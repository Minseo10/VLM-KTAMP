(define (problem blocksworld_pr4_4)
  (:domain blocksworld-original)
  (:objects
    blue grey magenta green
  )
  (:init
    (arm-empty)
    (on-table blue)
    (on grey blue)
    (on magenta grey)
    (on green magenta)
    (clear green)
  )
  (:goal
    (and
      (on-table green)
      (on magenta green)
      (on grey magenta)
      (on blue grey)
    )
  )
)