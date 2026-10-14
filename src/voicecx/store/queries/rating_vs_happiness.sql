-- 별점별 평균 행복·스트레스 지수 (지수 타당성 점검)
SELECT CAST(src.rating AS INTEGER) AS rating,
       COUNT(DISTINCT s.session_id) AS n_reviews,
       ROUND(AVG(CASE WHEN i.index_name = 'happiness' THEN i.raw END), 1) AS happiness,
       ROUND(AVG(CASE WHEN i.index_name = 'stress' THEN i.raw END), 1) AS stress,
       ROUND(AVG(CASE WHEN i.index_name = 'frustration' THEN i.raw END), 1) AS frustration
FROM sessions s
JOIN sources src ON src.source_id = s.source_id
JOIN indices i ON i.session_id = s.session_id
WHERE src.rating IS NOT NULL
GROUP BY CAST(src.rating AS INTEGER)
ORDER BY rating;
