-- 영상별 유형·크리에이터 지수·시청자 반응 (총평 가중 Valence 순)
SELECT s.source_id AS video_id,
       src.title,
       json_extract(src.meta_json, '$.video_type_label') AS video_type,
       json_extract(src.meta_json, '$.weighted_valence') AS weighted_valence,
       MAX(CASE WHEN i.index_name = 'happiness' THEN i.raw END) AS happiness,
       MAX(CASE WHEN i.index_name = 'stress' THEN i.raw END) AS stress,
       MAX(CASE WHEN i.index_name = 'engagement' THEN i.raw END) AS engagement,
       json_extract(src.meta_json, '$.creator_vs_audience.audience_minus_creator') AS audience_minus_creator,
       json_extract(src.meta_json, '$.n_comments') AS n_comments,
       json_extract(src.meta_json, '$.n_timestamp_comments') AS n_timestamp_comments
FROM sessions s
JOIN sources src ON src.source_id = s.source_id
JOIN indices i ON i.session_id = s.session_id
WHERE s.source_type = 'video'
GROUP BY s.session_id
ORDER BY weighted_valence DESC;
