-- 고객 여정 단계 × 감정 (절 단위). n이 작은 칸은 해석에 주의
SELECT u.journey_stage,
       COUNT(*) AS n_units,
       ROUND(AVG(u.valence), 3) AS mean_valence,
       ROUND(AVG(json_extract(u.scores_json, '$.stress')) * 100, 1) AS stress,
       ROUND(AVG(json_extract(u.scores_json, '$.frustration')) * 100, 1) AS frustration,
       ROUND(AVG(json_extract(u.scores_json, '$.happiness')) * 100, 1) AS happiness
FROM units u
WHERE u.journey_stage IS NOT NULL
GROUP BY u.journey_stage
ORDER BY mean_valence;
