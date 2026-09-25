import pandas as pd
from entity_resolution.normalization.normalize import normalize_df


def test_normalizer_preserves_nulls_and_transliterates():
    raw = pd.DataFrame({'entity_id': ['S1-1', 'S1-2'],
                        'business_name': ['Café Ltd.', 'Nan Inc.'],
                        'business_address': ['', '12 Main Road'],
                        'country': ['US', 'India']})
    out = normalize_df(raw)
    assert out.loc[0, 'name_norm'] == 'cafe'
    assert pd.isna(out.loc[0, 'address_norm'])
    assert out.loc[1, 'name_norm'] == 'nan'
    assert out.loc[1, 'legal_suffix'] == 'inc'
