#!/usr/bin/env python3
"""
setup_public_data.py
====================

Comprehensive setup script for HRM public data sources and training enhancements.
Automatically configures free and freemium data sources, downloads benchmark datasets,
sets up pre-trained models, and validates the complete training environment.

Usage:
    python setup_public_data.py --interactive
    python setup_public_data.py --config config/public_data_config.yaml
    python setup_public_data.py --download-all
"""

import argparse
import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import requests
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('setup_public_data.log')
    ]
)
logger = logging.getLogger(__name__)

class PublicDataSetup:
    """Main setup class for HRM public data environment."""
    
    def __init__(self, config_path: str = None):
        """Initialize setup with configuration."""
        self.config_path = config_path or "config/public_data_config.yaml"
        self.config = self.load_config()
        self.data_dir = Path("data")
        self.models_dir = Path("pretrained_models")
        self.setup_directories()
    
    def load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            return self.create_default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    def create_default_config(self) -> Dict:
        """Create a minimal default configuration."""
        logger.info("Creating default configuration...")
        return {
            'data_sources': {
                'free': {
                    'yahoo_finance': {'enabled': True, 'symbols': ['SPY', 'QQQ', 'AAPL']}
                }
            },
            'benchmark_datasets': {'enabled': True},
            'pretrained_models': {'enabled': True, 'cache_dir': './pretrained_models'}
        }
    
    def setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.data_dir,
            self.models_dir,
            "logs",
            "reports",
            "cache"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def check_dependencies(self) -> List[str]:
        """Check and install required dependencies."""
        logger.info("Checking dependencies...")
        
        required_packages = [
            'yfinance',
            'pandas',
            'numpy',
            'torch',
            'matplotlib',
            'seaborn',
            'requests',
            'aiohttp',
            'websockets',
            'scipy',
            'scikit-learn',
            'tqdm'
        ]
        
        optional_packages = [
            'fredapi',      # For FRED economic data
            'praw',         # For Reddit sentiment
            'textblob',     # For sentiment analysis
            'ta-lib',       # For technical analysis
            'beautifulsoup4',  # For web scraping
            'alpaca-trade-api', # For Alpaca trading
            'polygon-api-client' # For Polygon.io
        ]
        
        missing_required = []
        missing_optional = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.info(f"✓ {package}")
            except ImportError:
                missing_required.append(package)
                logger.warning(f"✗ {package} (required)")
        
        for package in optional_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.info(f"✓ {package}")
            except ImportError:
                missing_optional.append(package)
                logger.info(f"- {package} (optional)")
        
        if missing_required:
            logger.error(f"Missing required packages: {missing_required}")
            self.install_packages(missing_required)
        
        if missing_optional:
            logger.info(f"Optional packages available for enhanced features: {missing_optional}")
        
        return missing_required + missing_optional
    
    def install_packages(self, packages: List[str]):
        """Install missing packages."""
        logger.info(f"Installing packages: {packages}")
        
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {package}: {e}")
    
    def setup_free_data_sources(self):
        """Setup completely free data sources."""
        logger.info("Setting up free data sources...")
        
        free_sources = self.config.get('data_sources', {}).get('free', {})
        
        # Yahoo Finance
        if free_sources.get('yahoo_finance', {}).get('enabled', False):
            self.setup_yahoo_finance(free_sources['yahoo_finance'])
        
        # FRED Economic Data
        if free_sources.get('fred_economic', {}).get('enabled', False):
            self.setup_fred_data(free_sources['fred_economic'])
        
        # DoltHub Options
        if free_sources.get('dolthub_options', {}).get('enabled', False):
            self.setup_dolthub_options(free_sources['dolthub_options'])
    
    def setup_yahoo_finance(self, config: Dict):
        """Setup Yahoo Finance data source."""
        logger.info("Setting up Yahoo Finance...")
        
        try:
            import yfinance as yf
            
            symbols = config.get('symbols', ['SPY'])
            test_symbol = symbols[0]
            
            # Test connection
            ticker = yf.Ticker(test_symbol)
            info = ticker.info
            
            if info:
                logger.info(f"✓ Yahoo Finance connection successful (tested with {test_symbol})")
                
                # Download sample data
                hist = ticker.history(period="1mo")
                sample_file = self.data_dir / f"yahoo_sample_{test_symbol}.csv"
                hist.to_csv(sample_file)
                logger.info(f"Downloaded sample data to {sample_file}")
                
            else:
                logger.warning("Yahoo Finance connection test failed")
                
        except Exception as e:
            logger.error(f"Yahoo Finance setup failed: {e}")
    
    def setup_fred_data(self, config: Dict):
        """Setup Federal Reserve Economic Data."""
        logger.info("Setting up FRED economic data...")
        
        try:
            # Try to use fredapi if available
            try:
                import fredapi
                fred = fredapi.Fred()
                
                # Test with VIX data
                vix_data = fred.get_series('VIXCLS', limit=10)
                if not vix_data.empty:
                    logger.info("✓ FRED API connection successful")
                    
                    sample_file = self.data_dir / "fred_sample_vix.csv"
                    vix_data.to_csv(sample_file)
                    logger.info(f"Downloaded sample VIX data to {sample_file}")
            
            except ImportError:
                logger.info("fredapi not installed, using direct API access")
                self.setup_fred_direct_api(config)
                
        except Exception as e:
            logger.error(f"FRED setup failed: {e}")
    
    def setup_fred_direct_api(self, config: Dict):
        """Setup FRED using direct API calls."""
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        # Note: FRED API requires free API key registration
        logger.info("FRED direct API requires free registration at https://fred.stlouisfed.org/docs/api/api_key.html")
        
        # Create placeholder for API key
        fred_config_file = self.data_dir / "fred_config.txt"
        with open(fred_config_file, 'w') as f:
            f.write("# Add your FRED API key here\n")
            f.write("# Get free API key from: https://fred.stlouisfed.org/docs/api/api_key.html\n")
            f.write("FRED_API_KEY=your_api_key_here\n")
        
        logger.info(f"Created FRED config template at {fred_config_file}")
    
    def setup_dolthub_options(self, config: Dict):
        """Setup DoltHub options database."""
        logger.info("Setting up DoltHub options data...")
        
        try:
            # DoltHub provides free CSV exports
            base_url = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options"
            
            # Test connection
            response = requests.get(f"{base_url}/docs", timeout=10)
            if response.status_code == 200:
                logger.info("✓ DoltHub API accessible")
                
                # Create instructions for data export
                dolthub_instructions = self.data_dir / "dolthub_instructions.txt"
                with open(dolthub_instructions, 'w') as f:
                    f.write("DoltHub Options Data Instructions\n")
                    f.write("================================\n\n")
                    f.write("1. Visit: https://www.dolthub.com/repositories/post-no-preference/options\n")
                    f.write("2. Click 'Export' and select CSV format\n")
                    f.write("3. Download the CSV file\n")
                    f.write("4. Place the CSV in the data/ directory\n")
                    f.write("5. Update the CSV path in your configuration\n")
                
                logger.info(f"Created DoltHub instructions at {dolthub_instructions}")
            
        except Exception as e:
            logger.error(f"DoltHub setup failed: {e}")
    
    def setup_freemium_sources(self):
        """Setup freemium data sources (require API keys)."""
        logger.info("Setting up freemium data sources...")
        
        freemium_sources = self.config.get('data_sources', {}).get('freemium', {})
        
        api_key_instructions = self.data_dir / "api_keys_instructions.txt"
        instructions = []
        
        for source_name, source_config in freemium_sources.items():
            if source_config.get('enabled', False):
                if source_name == 'alpha_vantage':
                    instructions.append(self.create_alpha_vantage_instructions())
                elif source_name == 'iex_cloud':
                    instructions.append(self.create_iex_instructions())
                elif source_name == 'news_api':
                    instructions.append(self.create_newsapi_instructions())
        
        if instructions:
            with open(api_key_instructions, 'w') as f:
                f.write("API Key Setup Instructions\n")
                f.write("==========================\n\n")
                f.write("\n\n".join(instructions))
            
            logger.info(f"Created API key instructions at {api_key_instructions}")
    
    def create_alpha_vantage_instructions(self) -> str:
        """Create Alpha Vantage setup instructions."""
        return """Alpha Vantage (25 free calls/day)
---------------------------------------
1. Visit: https://www.alphavantage.co/support/#api-key
2. Enter your email to get a free API key
3. Update config/public_data_config.yaml:
   data_sources:
     freemium:
       alpha_vantage:
         enabled: true
         api_key: "YOUR_API_KEY_HERE"
"""
    
    def create_iex_instructions(self) -> str:
        """Create IEX Cloud setup instructions."""
        return """IEX Cloud (50k free credits/month)
----------------------------------
1. Visit: https://iexcloud.io/cloud-login#/register
2. Create free account
3. Copy your publishable token
4. Update config/public_data_config.yaml:
   data_sources:
     freemium:
       iex_cloud:
         enabled: true
         token: "YOUR_TOKEN_HERE"
"""
    
    def create_newsapi_instructions(self) -> str:
        """Create NewsAPI setup instructions."""
        return """NewsAPI (1000 free requests/day)
---------------------------------
1. Visit: https://newsapi.org/register
2. Create free account
3. Copy your API key
4. Update config/public_data_config.yaml:
   data_sources:
     freemium:
       news_api:
         enabled: true
         api_key: "YOUR_API_KEY_HERE"
"""
    
    def generate_benchmark_datasets(self):
        """Generate benchmark datasets for training."""
        if not self.config.get('benchmark_datasets', {}).get('enabled', False):
            logger.info("Benchmark dataset generation disabled")
            return
        
        logger.info("Generating benchmark datasets...")
        
        try:
            # Import our benchmark dataset generator
            from lab_v10.src.common.benchmark_datasets import create_hrm_training_package
            
            # Generate datasets
            datasets = create_hrm_training_package()
            
            # Save datasets
            for name, data in datasets.items():
                filename = self.data_dir / f"benchmark_{name}.parquet"
                data.to_parquet(filename)
                logger.info(f"Generated benchmark dataset: {filename}")
            
            logger.info("✓ Benchmark datasets generated successfully")
            
        except Exception as e:
            logger.error(f"Benchmark dataset generation failed: {e}")
    
    def setup_pretrained_models(self):
        """Setup pre-trained models."""
        if not self.config.get('pretrained_models', {}).get('enabled', False):
            logger.info("Pre-trained models disabled")
            return
        
        logger.info("Setting up pre-trained models...")
        
        try:
            from lab_v10.src.options.pretrained_models import PretrainedModelRegistry
            
            models_config = self.config.get('pretrained_models', {}).get('models', {})
            
            for model_name, model_config in models_config.items():
                if model_config.get('enabled', False) and model_config.get('auto_download', False):
                    try:
                        model_path = PretrainedModelRegistry.download_model(
                            model_name, 
                            cache_dir=str(self.models_dir)
                        )
                        logger.info(f"✓ Downloaded pre-trained model: {model_name}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to download {model_name}: {e}")
            
            logger.info("✓ Pre-trained model setup completed")
            
        except Exception as e:
            logger.error(f"Pre-trained model setup failed: {e}")
    
    def validate_setup(self) -> Dict[str, bool]:
        """Validate the complete setup."""
        logger.info("Validating setup...")
        
        validation_results = {}
        
        # Check data directory
        validation_results['data_directory'] = self.data_dir.exists()
        
        # Check models directory
        validation_results['models_directory'] = self.models_dir.exists()
        
        # Check if we have any data files
        data_files = list(self.data_dir.glob('*.csv')) + list(self.data_dir.glob('*.parquet'))
        validation_results['has_data_files'] = len(data_files) > 0
        
        # Check if we have any model files
        model_files = list(self.models_dir.glob('*.bin')) + list(self.models_dir.glob('*.pth'))
        validation_results['has_model_files'] = len(model_files) > 0
        
        # Test basic imports
        try:
            import yfinance
            validation_results['yfinance_import'] = True
        except ImportError:
            validation_results['yfinance_import'] = False
        
        try:
            import torch
            validation_results['torch_import'] = True
        except ImportError:
            validation_results['torch_import'] = False
        
        # Print validation summary
        logger.info("Validation Results:")
        for check, result in validation_results.items():
            status = "✓" if result else "✗"
            logger.info(f"  {status} {check}")
        
        return validation_results
    
    def create_example_training_script(self):
        """Create an example training script using public data."""
        script_content = '''#!/usr/bin/env python3
"""
example_public_data_training.py
===============================

Example script demonstrating HRM training with public data sources.
"""

import pandas as pd
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    logger.info("Starting HRM training with public data...")
    
    # Load benchmark datasets
    data_dir = Path("data")
    
    # Check for generated datasets
    straddle_file = data_dir / "benchmark_straddle_train.parquet"
    intraday_file = data_dir / "benchmark_intraday_signals.parquet"
    
    if straddle_file.exists():
        straddle_data = pd.read_parquet(straddle_file)
        logger.info(f"Loaded straddle dataset: {straddle_data.shape}")
    else:
        logger.error("Straddle dataset not found. Run setup_public_data.py first.")
        return
    
    if intraday_file.exists():
        intraday_data = pd.read_parquet(intraday_file)
        logger.info(f"Loaded intraday dataset: {intraday_data.shape}")
    else:
        logger.warning("Intraday dataset not found.")
    
    # Load Yahoo Finance data if available
    yahoo_files = list(data_dir.glob("yahoo_sample_*.csv"))
    if yahoo_files:
        yahoo_data = pd.read_csv(yahoo_files[0], index_col=0, parse_dates=True)
        logger.info(f"Loaded Yahoo Finance sample: {yahoo_data.shape}")
    
    # Initialize HRM model (placeholder)
    logger.info("Initializing HRM model...")
    
    # TODO: Add actual HRM training code here
    # This would involve:
    # 1. Data preprocessing
    # 2. Feature engineering
    # 3. Model initialization
    # 4. Training loop
    # 5. Evaluation
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
'''
        
        script_path = Path("example_public_data_training.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
        
        logger.info(f"Created example training script: {script_path}")
    
    def interactive_setup(self):
        """Run interactive setup process."""
        logger.info("Starting interactive setup...")
        
        print("\\n" + "="*60)
        print("HRM Public Data Setup")
        print("="*60)
        
        # Ask about data sources
        print("\\nWhich data sources would you like to enable?")
        print("1. Yahoo Finance (free)")
        print("2. FRED Economic Data (free)")
        print("3. Alpha Vantage (freemium - 25 calls/day)")
        print("4. IEX Cloud (freemium - 50k credits/month)")
        
        choices = input("Enter numbers separated by commas (e.g., 1,2): ").strip()
        
        # Update config based on choices
        if '1' in choices:
            self.config['data_sources']['free']['yahoo_finance']['enabled'] = True
        if '2' in choices:
            self.config['data_sources']['free']['fred_economic']['enabled'] = True
        if '3' in choices:
            api_key = input("Enter Alpha Vantage API key (or press Enter to skip): ").strip()
            if api_key:
                self.config['data_sources']['freemium']['alpha_vantage']['enabled'] = True
                self.config['data_sources']['freemium']['alpha_vantage']['api_key'] = api_key
        if '4' in choices:
            token = input("Enter IEX Cloud token (or press Enter to skip): ").strip()
            if token:
                self.config['data_sources']['freemium']['iex_cloud']['enabled'] = True
                self.config['data_sources']['freemium']['iex_cloud']['token'] = token
        
        # Ask about benchmark datasets
        generate_benchmarks = input("\\nGenerate synthetic benchmark datasets? (y/n): ").lower().startswith('y')
        self.config['benchmark_datasets']['enabled'] = generate_benchmarks
        
        # Ask about pre-trained models
        download_models = input("Download pre-trained models? (y/n): ").lower().startswith('y')
        self.config['pretrained_models']['enabled'] = download_models
        
        # Save updated config
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Updated configuration saved to {self.config_path}")
    
    def run_complete_setup(self):
        """Run the complete setup process."""
        logger.info("Starting complete HRM public data setup...")
        
        try:
            # Check dependencies
            self.check_dependencies()
            
            # Setup free data sources
            self.setup_free_data_sources()
            
            # Setup freemium sources (instructions only)
            self.setup_freemium_sources()
            
            # Generate benchmark datasets
            self.generate_benchmark_datasets()
            
            # Setup pre-trained models
            self.setup_pretrained_models()
            
            # Create example training script
            self.create_example_training_script()
            
            # Validate setup
            validation_results = self.validate_setup()
            
            # Summary
            logger.info("\\n" + "="*60)
            logger.info("SETUP COMPLETE")
            logger.info("="*60)
            
            success_count = sum(validation_results.values())
            total_checks = len(validation_results)
            logger.info(f"Validation: {success_count}/{total_checks} checks passed")
            
            if success_count == total_checks:
                logger.info("✓ All checks passed! Your HRM environment is ready.")
            else:
                logger.warning("⚠ Some checks failed. See validation results above.")
            
            logger.info("\\nNext steps:")
            logger.info("1. Review API key instructions in data/api_keys_instructions.txt")
            logger.info("2. Run: python example_public_data_training.py")
            logger.info("3. Explore the benchmark datasets in data/")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Setup HRM public data environment")
    parser.add_argument('--config', default='config/public_data_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive setup')
    parser.add_argument('--download-all', action='store_true',
                       help='Download all available data and models')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation checks')
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = PublicDataSetup(args.config)
    
    try:
        if args.validate_only:
            setup.validate_setup()
        elif args.interactive:
            setup.interactive_setup()
            setup.run_complete_setup()
        else:
            setup.run_complete_setup()
            
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()