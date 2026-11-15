"""Species lookup and nearest-neighbour matching."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import logging

logger = logging.getLogger(__name__)


class SpeciesDatabase:
    def __init__(self, data_path: str = "data/raw/filovirus_data_template.csv"):
        self.data_path = Path(data_path)
        self.df = None
        self.species_features = {}
        self.scaler = StandardScaler()
        self.feature_columns = [
            'viral_genome_length', 'gc_content', 'cpg_frequency',
            'glycoprotein_length', 'gp_hydrophobicity',
            'body_mass_kg', 'lifespan_years', 'geographic_range_km2'
        ]
        self.categorical_columns = [
            'diet_type', 'social_behavior', 'habitat_type', 'human_contact_frequency'
        ]

        self._load_database()

    def _load_database(self):
        logger.info(f"Loading species database from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} records with {self.df['host_species'].nunique()} species")

        self._compute_species_profiles()

        # distances are meaningless unless the columns are on the same scale
        self.scaler.fit(self.df[self.feature_columns])
        logger.info("Species database initialized successfully")

    def _compute_species_profiles(self):
        """One row per species: numeric columns averaged, categoricals by mode."""
        logger.info("Computing species profiles...")

        for species in self.df['host_species'].unique():
            species_data = self.df[self.df['host_species'] == species]

            numerical_features = {
                col: species_data[col].mean()
                for col in self.feature_columns
            }

            categorical_features = {
                col: species_data[col].mode()[0] if len(species_data[col].mode()) > 0 else species_data[col].iloc[0]
                for col in self.categorical_columns
            }

            taxonomic_info = {
                'genus': species.split('_')[0],
                'order': species_data['host_order'].mode()[0] if 'host_order' in species_data.columns else 'Unknown',
                'family': species_data['host_family'].mode()[0] if 'host_family' in species_data.columns else 'Unknown',
                'infection_rate': species_data['infection_status'].mean() if 'infection_status' in species_data.columns else 0.5,
                'sample_count': len(species_data),
            }

            self.species_features[species] = {
                **numerical_features,
                **categorical_features,
                **taxonomic_info
            }

        logger.info(f"Computed profiles for {len(self.species_features)} species")

    def get_species_list(self) -> List[str]:
        return sorted(self.species_features.keys())

    def get_species_info(self, species_name: str) -> Optional[Dict]:
        return self.species_features.get(species_name)

    def species_exists(self, species_name: str) -> bool:
        return species_name in self.species_features

    def find_similar_species(self, species_name: str, n_neighbors: int = 5) -> List[Tuple[str, float]]:
        """(species, similarity) sorted best first; similarity is 1/(1+euclidean), so 1.0 is identical."""
        if not self.species_exists(species_name):
            return []

        target_features = self.species_features[species_name]
        target_vector = self._extract_feature_vector(target_features)

        similarities = []
        for other_species, other_features in self.species_features.items():
            if other_species == species_name:
                continue

            other_vector = self._extract_feature_vector(other_features)
            distance = euclidean_distances([target_vector], [other_vector])[0][0]
            similarity = 1 / (1 + distance)

            similarities.append((other_species, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_neighbors]

    def predict_features_for_unknown_species(
        self,
        species_name: str,
        n_neighbors: int = 5,
        weights: str = 'distance'
    ) -> Tuple[Dict, float]:
        """Returns (features, confidence 0-1). Same-genus matches are trusted at a flat 0.85."""
        logger.info(f"Predicting features for unknown species: {species_name}")

        parts = species_name.split('_')
        genus = parts[0]

        # same genus first - usually a much better guess than the global neighbours
        same_genus = [s for s in self.species_features.keys() if s.startswith(genus + "_")]

        if same_genus:
            logger.info(f"Found {len(same_genus)} species with same genus")
            return self._interpolate_features(same_genus, n_neighbors, weights), 0.85

        logger.info("Using global nearest neighbor approach")
        target_vector = self._extract_generic_vector(species_name)

        similarities = []
        for other_species, other_features in self.species_features.items():
            other_vector = self._extract_feature_vector(other_features)
            distance = euclidean_distances([target_vector], [other_vector])[0][0]
            similarity = 1 / (1 + distance)
            similarities.append((other_species, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_species = [s[0] for s in similarities[:n_neighbors]]

        top_similarities = [s[1] for s in similarities[:n_neighbors]]
        confidence = np.mean(top_similarities)

        return self._interpolate_features(top_species, n_neighbors, weights), confidence

    def _extract_feature_vector(self, features_dict: Dict) -> np.ndarray:
        vector = np.array([
            features_dict[col] for col in self.feature_columns
        ])
        # wrap in a frame so the scaler sees the column names it was fitted with
        import pandas as pd
        df = pd.DataFrame([vector], columns=self.feature_columns)
        return self.scaler.transform(df)[0]

    def _extract_generic_vector(self, species_name: str) -> np.ndarray:
        """Crude guess from keywords in the name, for species we have nothing else on."""
        features = {
            'viral_genome_length': 18950,
            'gc_content': 41,
            'cpg_frequency': 0.038,
            'glycoprotein_length': 676,
            'gp_hydrophobicity': -0.45,
            'body_mass_kg': 1.0,
            'lifespan_years': 10,
            'geographic_range_km2': 1e6,
        }

        if any(x in species_name.lower() for x in ['bat', 'chiroptera']):
            features['body_mass_kg'] = 0.05
            features['lifespan_years'] = 20
            features['gc_content'] = 42
            features['geographic_range_km2'] = 1e7

        if any(x in species_name.lower() for x in ['sapiens', 'primate']):
            features['body_mass_kg'] = 70
            features['lifespan_years'] = 72
            features['geographic_range_km2'] = 5e7

        if any(x in species_name.lower() for x in ['rodent', 'mouse', 'rat']):
            features['body_mass_kg'] = 0.02
            features['lifespan_years'] = 3

        vector = np.array([features[col] for col in self.feature_columns])
        import pandas as pd
        df = pd.DataFrame([vector], columns=self.feature_columns)
        return self.scaler.transform(df)[0]

    def _interpolate_features(
        self,
        species_list: List[str],
        n_neighbors: int,
        weights: str = 'distance'
    ) -> Dict:
        selected_species = species_list[:n_neighbors] if len(species_list) >= n_neighbors else species_list

        # weights arg is accepted but everything is averaged evenly for now
        interpolated = {}

        for col in self.feature_columns:
            values = [self.species_features[s][col] for s in selected_species]
            interpolated[col] = np.mean(values)

        for col in self.categorical_columns:
            values = [self.species_features[s][col] for s in selected_species]
            # plain mode; scipy.stats.mode was fussy about dtypes here
            interpolated[col] = max(set(values), key=values.count)

        # taxonomy copied from the closest match rather than averaged
        template = self.species_features[selected_species[0]]
        interpolated['order'] = template['order']
        interpolated['family'] = template['family']
        interpolated['infection_rate'] = np.mean([
            self.species_features[s].get('infection_rate', 0.5) for s in selected_species
        ])

        return interpolated

    def get_species_by_order(self, order_name: str) -> List[str]:
        return [
            s for s, features in self.species_features.items()
            if features.get('order') == order_name
        ]

    def get_high_risk_species(self, threshold: float = 0.7) -> List[Tuple[str, float]]:
        high_risk = [
            (s, features['infection_rate'])
            for s, features in self.species_features.items()
            if features.get('infection_rate', 0) >= threshold
        ]
        return sorted(high_risk, key=lambda x: x[1], reverse=True)

    def get_statistics(self) -> Dict:
        infection_rates = [f['infection_rate'] for f in self.species_features.values()]
        sample_counts = [f['sample_count'] for f in self.species_features.values()]

        return {
            'total_species': len(self.species_features),
            'total_records': len(self.df),
            'avg_infection_rate': np.mean(infection_rates),
            'avg_samples_per_species': np.mean(sample_counts),
            'orders': len(set(f['order'] for f in self.species_features.values())),
            'families': len(set(f['family'] for f in self.species_features.values())),
        }


_db_instance = None


def get_species_database(data_path: str = "data/raw/filovirus_data_template.csv") -> SpeciesDatabase:
    global _db_instance
    if _db_instance is None:
        _db_instance = SpeciesDatabase(data_path)
    return _db_instance
