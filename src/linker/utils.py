NORMALIZER_MODEL_MAPPINGS = {
    "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large": "sapBERT-multilanguage-large",
    "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR": "sapBERT-multilanguage",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token": "sapBERT-english",
}

NORMALIZER_MODEL_MAPPINGS_REVERSED = {v:k for k,v in NORMALIZER_MODEL_MAPPINGS.items()}