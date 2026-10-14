-- 영상 구간별 크리에이터 감정과 그 구간을 가리킨 타임스탬프 댓글의 감정
SELECT s.source_id AS video_id,
       u.segment_index,
       u.segment,
       MIN(u.start_sec) AS start_sec,
       MAX(u.end_sec) AS end_sec,
       SUM(u.speaker_role != 'commenter') AS n_sentences,
       ROUND(AVG(CASE WHEN u.speaker_role != 'commenter' THEN u.valence END), 3) AS creator_valence,
       SUM(u.speaker_role = 'commenter') AS n_timestamp_comments,
       ROUND(AVG(CASE WHEN u.speaker_role = 'commenter' THEN u.valence END), 3) AS comment_valence
FROM units u
JOIN sessions s ON s.session_id = u.session_id
WHERE u.source_type = 'video' AND u.segment_index IS NOT NULL
GROUP BY s.source_id, u.segment_index
ORDER BY s.source_id, u.segment_index;
