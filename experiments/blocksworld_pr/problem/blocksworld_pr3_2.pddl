(define (problem blocksworld_pr3_2)
  (:domain blocksworld-original)
  (:objects
    green magenta brown
  )
  (:init
    (arm-empty)
    (on-table green)
    (on magenta green)
    (clear magenta)
    (on-table brown)
    (clear brown)
  )
  (:goal
    (and
      (on-table brown)
      (on magenta brown)
      (on-table green)
    )
  )
)