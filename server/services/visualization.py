import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from typing import Optional, Dict

matplotlib.use('Agg')

class VisualizationService:
    """Handles all plotting and visualization operations."""
    
    @staticmethod
    def isolate_feature_data(dataframe: pd.DataFrame, feature_prefix: str) -> pd.Series:
        """Extract and average feature data by prefix."""
        feature_cols = [col for col in dataframe.columns if col.startswith(feature_prefix)]
        feature_df = dataframe[feature_cols]
        feature_df.columns = range(1, len(feature_df.columns) + 1)
        return feature_df.mean(axis=0)
    
    @staticmethod
    def create_comparison_plot(
        uploaded_data: pd.Series,
        speaker_data: Optional[pd.Series] = None,
        ylabel: str = '',
        title: str = ''
    ) -> str:
        """
        Create comparison plot between uploaded and speaker data.
        
        Returns:
            Base64 encoded plot image
        """
        plt.figure(figsize=(12, 6))
        plt.plot(uploaded_data, linestyle='-', color='red', label='Uploaded')
        
        if speaker_data is not None:
            plt.plot(speaker_data, linestyle='-', color='green', label='Speaker')
        
        plt.xlabel('Feature Column')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convert to base64
        bytes_io = io.BytesIO()
        plt.savefig(bytes_io, format='jpg', dpi=100, bbox_inches='tight')
        bytes_io.seek(0)
        plt.close()  # Important: close figure to prevent memory leaks
        
        return base64.b64encode(bytes_io.read()).decode()
    
    def generate_feature_plots(
        self,
        uploaded_df: pd.DataFrame,
        speaker_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, str]:
        """Generate all feature comparison plots."""
        features = [
            ('MFCC', 'mean MFCC value', 'Mean MFCC'),
            ('RMS', 'mean RMS value', 'Mean RMS'),
            ('ZCR', 'mean ZCR value', 'Mean ZCR'),
            ('SpectralCentroid', 'mean Spectral Centroid value', 'Mean Spectral Centroid'),
            ('SpectralBandwidth', 'mean Spectral Bandwidth value', 'Mean Spectral Bandwidth')
        ]
        
        plots = {}
        for feature_name, ylabel, title in features:
            uploaded_data = self.isolate_feature_data(uploaded_df, feature_name)
            speaker_data = None
            
            if speaker_df is not None:
                speaker_data = self.isolate_feature_data(speaker_df, feature_name)
            
            plot_key = f"{feature_name.lower()}_plot"
            plots[plot_key] = self.create_comparison_plot(
                uploaded_data, speaker_data, ylabel, title
            )
        
        return plots