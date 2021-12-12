# MultiTask Model
including MMOE, SNR_trans, SNR_avg, PLE

# dataset
## Columns and Labels

```py
['clicked','followed','fan_count','upload_photo_cnt_7d', 'upload_photo_cnt_30d',
       'accu_public_visible_photo_cnt', 'combo_user_uv_1d',
       'combo_user_uv_7d', 'combo_user_click_1d', 'combo_user_click_7d',
       'combo_user_follow_1d', 'combo_user_follow_7d', 'usertab_user_uv_1d',
       'usertab_user_uv_7d', 'usertab_user_click_1d', 'usertab_user_click_7d',
       'usertab_user_follow_1d', 'usertab_user_follow_7d',
       'combo_keyword_uv_1d', 'combo_keyword_uv_7d', 'user_keyword_uv_1d',
       'user_keyword_uv_7d', 'combo_keyword_user_uv_1d',
       'combo_keyword_user_uv_7d', 'combo_keyword_user_click_1d',
       'combo_keyword_user_click_7d', 'combo_keyword_user_follow_1d',
       'combo_keyword_user_follow_7d', 'user_keyword_user_uv_1d',
       'user_keyword_user_uv_7d', 'user_keyword_user_click_1d',
       'user_keyword_user_click_7d', 'user_keyword_user_follow_1d',
       'user_keyword_user_follow_7d', 'relevance_score',
       'combo_keyword_user_ctr_1d', 'combo_keyword_user_ctr_7d',
       'combo_keyword_user_ftr_1d', 'combo_keyword_user_ftr_7d',
       'user_keyword_user_ctr_1d', 'user_keyword_user_ctr_7d',
       'user_keyword_user_ftr_1d', 'user_keyword_user_ftr_7d']
```

Tasks: predict whether a certain user would be `clicked` and `followed` under certain query. 


## Data Preprocessing
- Categorical feature: use one-hot encoding
- Numerical feature: discretize numerical features with equal-depths binning






