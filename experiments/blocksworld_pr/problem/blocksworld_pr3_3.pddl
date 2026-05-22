(define (problem blocksworld_pr3_3)
  (:domain blocksworld-original)
  (:objects
    white magenta brown
  )
  (:init
    (arm-empty)
    (on-table white)
    (on magenta white)
    (clear magenta)
    (on-table brown)
    (clear brown)
  )
  (:goal
    (and
      (on-table magenta)
      (on white magenta)
      (on-table brown)
    )
  )
)