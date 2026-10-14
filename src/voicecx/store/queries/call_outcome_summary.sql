-- 통화 결과(해결·보류·이관)별 평균 지수와 Peak-End
SELECT json_extract(src.meta_json, '$.outcome') AS outcome,
       COUNT(DISTINCT s.session_id) AS n_calls,
       ROUND(AVG(CASE WHEN i.index_name = 'frustration' THEN i.raw END), 1) AS frustration,
       ROUND(AVG(CASE WHEN i.index_name = 'recovery' THEN i.raw END), 1) AS recovery,
       ROUND(AVG(CASE WHEN i.index_name = 'happiness' THEN i.raw END), 1) AS happiness,
       ROUND(AVG(s.peak_end), 1) AS peak_end
FROM sessions s
JOIN sources src ON src.source_id = s.source_id
JOIN indices i ON i.session_id = s.session_id
WHERE s.source_type = 'call'
GROUP BY outcome
ORDER BY peak_end DESC;
