-- 통화별 결과·대화 역학·지수 (좌절이 높은 순)
SELECT s.source_id AS call_id,
       json_extract(src.meta_json, '$.outcome') AS outcome,
       json_extract(src.meta_json, '$.duration') AS duration_sec,
       json_extract(src.meta_json, '$.customer_interruptions') AS interruptions,
       json_extract(src.meta_json, '$.repeat_count') AS repeats,
       json_extract(src.meta_json, '$.max_silence') AS max_silence,
       MAX(CASE WHEN i.index_name = 'frustration' THEN i.raw END) AS frustration,
       MAX(CASE WHEN i.index_name = 'stress' THEN i.raw END) AS stress,
       MAX(CASE WHEN i.index_name = 'recovery' THEN i.raw END) AS recovery,
       s.peak_end,
       (SELECT COUNT(*) FROM events e WHERE e.session_id = s.session_id) AS n_events
FROM sessions s
JOIN sources src ON src.source_id = s.source_id
JOIN indices i ON i.session_id = s.session_id
WHERE s.source_type = 'call'
GROUP BY s.session_id
ORDER BY frustration DESC;
