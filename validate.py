"""
Validation script to check system setup and configuration.
Run this before building the index to catch issues early.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def check_imports():
    """Verify all required packages are installed."""
    logger.info("Checking imports...")
    
    missing = []
    packages = {
        "torch": "PyTorch",
        "faiss": "FAISS",
        "sentence_transformers": "sentence-transformers",
        "psycopg2": "psycopg2",
        "numpy": "NumPy",
        "dotenv": "python-dotenv"
    }
    
    for module, name in packages.items():
        try:
            __import__(module)
            logger.info(f"  ✓ {name}")
        except ImportError:
            logger.error(f"  ✗ {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.error("Install with: pip install -r requirements.txt")
        return False
    
    return True


def check_cuda():
    """Check CUDA/GPU availability."""
    logger.info("\nChecking CUDA...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"  ✓ CUDA available")
            logger.info(f"    Device: {device_name}")
            logger.info(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            logger.warning("  ⚠ CUDA not available - will use CPU (slower)")
            return True  # Not fatal
    except Exception as e:
        logger.error(f"  ✗ Error checking CUDA: {e}")
        return False


def check_database():
    """Check database connection."""
    logger.info("\nChecking database connection...")
    
    try:
        from config import DatabaseConfig
        import psycopg2
        
        db_config = DatabaseConfig.from_env()
        
        try:
            db_config.validate()
        except ValueError as e:
            logger.error(f"  ✗ {e}")
            logger.error("    Please set environment variables in .env file:")
            logger.error("    DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME")
            return False
        
        # Try to connect
        try:
            conn = psycopg2.connect(
                user=db_config.user,
                password=db_config.password,
                host=db_config.host,
                port=db_config.port,
                dbname=db_config.dbname,
            )
            logger.info(f"  ✓ Connected to {db_config.host}:{db_config.port}/{db_config.dbname}")
            
            # Check if reviews table exists
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'reviews'
                    );
                """)
                table_exists = cur.fetchone()[0]
                
                if table_exists:
                    # Count rows
                    cur.execute('SELECT COUNT(*) FROM reviews;')
                    count = cur.fetchone()[0]
                    logger.info(f"  ✓ Table 'reviews' exists with {count:,} rows")
                else:
                    logger.error("  ✗ Table 'reviews' not found")
                    conn.close()
                    return False
            
            conn.close()
            return True
            
        except psycopg2.Error as e:
            logger.error(f"  ✗ Database connection failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        return False


def check_model():
    """Check if model can be loaded."""
    logger.info("\nChecking embedding model...")
    
    try:
        from config import ModelConfig
        from sentence_transformers import SentenceTransformer
        import torch
        
        model_config = ModelConfig()
        logger.info(f"  Loading {model_config.name}...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        
        model = SentenceTransformer(
            model_config.name,
            trust_remote_code=True,
            device=device,
            model_kwargs={"dtype": dtype_map.get(model_config.dtype, torch.float16)},
        )
        
        embedding_dim = model.get_sentence_embedding_dimension()
        logger.info(f"  ✓ Model loaded successfully")
        logger.info(f"    Embedding dimension: {embedding_dim}")
        logger.info(f"    Device: {device}")
        logger.info(f"    Max sequence length: {model_config.max_seq_length}")
        
        # Test encoding
        logger.info("  Testing encoding...")
        test_embedding = model.encode("test", normalize_embeddings=True)
        logger.info(f"  ✓ Encoding works (shape: {test_embedding.shape})")
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Model loading failed: {e}")
        return False


def check_directories():
    """Check that required directories exist or can be created."""
    logger.info("\nChecking directories...")
    
    try:
        from config import IndexConfig
        
        index_config = IndexConfig()
        
        if not index_config.index_dir.exists():
            index_config.index_dir.mkdir(parents=True)
            logger.info(f"  ✓ Created directory: {index_config.index_dir}")
        else:
            logger.info(f"  ✓ Directory exists: {index_config.index_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Directory check failed: {e}")
        return False


def check_existing_index():
    """Check if an index already exists."""
    logger.info("\nChecking for existing index...")
    
    try:
        from config import ModelConfig, IndexConfig
        
        model_config = ModelConfig()
        index_config = IndexConfig()
        
        index_path = index_config.get_index_path(model_config.name)
        meta_path = index_config.get_metadata_path(model_config.name)
        
        if index_path.exists() and meta_path.exists():
            logger.warning(f"  ⚠ Index already exists at {index_path}")
            logger.warning("    Running indexer.py will overwrite it")
            
            # Try to load and show info
            try:
                import faiss
                import pickle
                
                index = faiss.read_index(str(index_path))
                with open(meta_path, "rb") as f:
                    doc_ids = pickle.load(f)
                
                logger.info(f"    Existing index has {index.ntotal:,} vectors")
                logger.info(f"    Metadata has {len(doc_ids):,} document IDs")
            except Exception as e:
                logger.warning(f"    Could not read existing index: {e}")
        else:
            logger.info("  ✓ No existing index found (ready for fresh build)")
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Error checking index: {e}")
        return False


def main():
    """Run all validation checks."""
    logger.info("="*60)
    logger.info("RAG System Validation")
    logger.info("="*60)
    
    checks = [
        ("Package imports", check_imports),
        ("CUDA/GPU", check_cuda),
        ("Database", check_database),
        ("Embedding model", check_model),
        ("Directories", check_directories),
        ("Existing index", check_existing_index),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            logger.error(f"\nUnexpected error in {name}: {e}")
            results[name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Validation Summary")
    logger.info("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status}: {name}")
    
    logger.info(f"\nResult: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("\n✓ System validation successful!")
        logger.info("You can now run: python indexer.py")
        return 0
    else:
        logger.error("\n✗ System validation failed!")
        logger.error("Please fix the issues above before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())