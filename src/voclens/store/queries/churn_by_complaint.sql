-- 불만 유형별 이탈 의사·재주문 의사 비율 (리뷰 단위)
WITH review_signals AS (
    SELECT u.session_id,
           MAX(COALESCE(json_extract(u.cx_signals_json, '$.churn_intent'), 0)) AS churn,
           MAX(COALESCE(json_extract(u.cx_signals_json, '$.repurchase_intent'), 0)) AS repurchase
    FROM units u
    GROUP BY u.session_id
),
review_labels AS (
    SELECT DISTINCT u.session_id, l.label
    FROM unit_labels l JOIN units u ON u.unit_id = l.unit_id
    WHERE l.label_type = 'complaint'
)
SELECT rl.label,
       COUNT(*) AS n_reviews,
       ROUND(100.0 * AVG(rs.churn), 1) AS churn_pct,
       ROUND(100.0 * AVG(rs.repurchase), 1) AS repurchase_pct
FROM review_labels rl JOIN review_signals rs ON rs.session_id = rl.session_id
GROUP BY rl.label
ORDER BY churn_pct DESC, n_reviews DESC;
