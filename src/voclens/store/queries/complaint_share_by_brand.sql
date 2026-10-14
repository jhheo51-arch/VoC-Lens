-- 브랜드별 불만 유형 비율: 해당 유형을 언급한 리뷰 수 / 브랜드 전체 리뷰 수
WITH totals AS (
    SELECT src.brand, COUNT(*) AS n_reviews
    FROM sessions s JOIN sources src ON src.source_id = s.source_id
    WHERE s.source_type = 'review'
    GROUP BY src.brand
),
mentions AS (
    SELECT src.brand, l.category, l.label, COUNT(DISTINCT u.session_id) AS n_mentions
    FROM unit_labels l
    JOIN units u ON u.unit_id = l.unit_id
    JOIN sessions s ON s.session_id = u.session_id
    JOIN sources src ON src.source_id = s.source_id
    WHERE l.label_type = 'complaint' AND l.category IS NOT NULL AND s.source_type = 'review'
    GROUP BY src.brand, l.category, l.label
)
SELECT m.brand, m.category, m.label, m.n_mentions, t.n_reviews,
       ROUND(100.0 * m.n_mentions / t.n_reviews, 1) AS share_pct
FROM mentions m JOIN totals t ON t.brand IS m.brand
ORDER BY m.brand, share_pct DESC;
