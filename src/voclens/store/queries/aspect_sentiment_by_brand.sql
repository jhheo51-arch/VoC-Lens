-- 브랜드 × 속성별 평균 감성 (-1 ~ 1)과 긍정·부정 절 수
SELECT src.brand, u.aspect,
       COUNT(*) AS n_units,
       ROUND(AVG(u.aspect_sentiment), 3) AS mean_sentiment,
       SUM(u.aspect_sentiment > 0.1) AS n_positive,
       SUM(u.aspect_sentiment < -0.1) AS n_negative
FROM units u
JOIN sessions s ON s.session_id = u.session_id
JOIN sources src ON src.source_id = s.source_id
WHERE u.aspect IS NOT NULL
GROUP BY src.brand, u.aspect
ORDER BY src.brand, mean_sentiment;
