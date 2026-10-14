-- 리뷰 이벤트 참여 리뷰와 일반 리뷰의 지수·별점 차이 (긍정 편향 점검)
SELECT CASE WHEN instr(s.flags_json, 'review_event') > 0 THEN '리뷰 이벤트' ELSE '일반' END AS review_type,
       COUNT(DISTINCT s.session_id) AS n_reviews,
       ROUND(AVG(src.rating), 2) AS mean_rating,
       ROUND(AVG(CASE WHEN i.index_name = 'happiness' THEN i.raw END), 1) AS happiness,
       ROUND(AVG(CASE WHEN i.index_name = 'stress' THEN i.raw END), 1) AS stress
FROM sessions s
JOIN sources src ON src.source_id = s.source_id
JOIN indices i ON i.session_id = s.session_id
WHERE s.source_type = 'review'
GROUP BY review_type;
