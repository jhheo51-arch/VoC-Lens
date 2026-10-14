-- 스트레스·좌절이 가장 높은 절 20개 (근거 원문 확인용)
SELECT src.brand, u.text, u.journey_stage,
       ROUND(json_extract(u.scores_json, '$.stress') * 100, 1) AS stress,
       ROUND(json_extract(u.scores_json, '$.frustration') * 100, 1) AS frustration,
       (SELECT group_concat(l.label, ', ') FROM unit_labels l
         WHERE l.unit_id = u.unit_id AND l.label_type = 'complaint') AS complaints
FROM units u
JOIN sessions s ON s.session_id = u.session_id
JOIN sources src ON src.source_id = s.source_id
ORDER BY json_extract(u.scores_json, '$.stress') + json_extract(u.scores_json, '$.frustration') DESC
LIMIT 20;
