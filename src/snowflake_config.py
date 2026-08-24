# ============================================================
# Snowflake Configuration for Facial Emotion Recognition System
# Author: Corey Leath (CoreyLeath-code)
# ============================================================

import os

SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "ML_WH"),
    "database": os.getenv("SNOWFLAKE_DATABASE", "MLDB"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
}
