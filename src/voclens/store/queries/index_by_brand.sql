-- 채널·브랜드별 6개 지수 평균과 표본 수
SELECT s.source_type, src.brand, i.index_name,
       COUNT(i.raw) AS n,
       ROUND(AVG(i.raw), 1) AS mean,
       ROUND(MIN(i.raw), 1) AS min,
       ROUND(MAX(i.raw), 1) AS max
FROM indices i
JOIN sessions s ON s.session_id = i.session_id
JOIN sources src ON src.source_id = s.source_id
WHERE i.raw IS NOT NULL
GROUP BY s.source_type, src.brand, i.index_name
ORDER BY s.source_type, src.brand, i.index_name;
