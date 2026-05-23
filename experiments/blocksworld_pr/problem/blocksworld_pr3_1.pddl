(define (problem blocksworld_pr3_1)
  (:domain blocksworld-original)
  (:objects
    green red grey
  )
  (:init
    (arm-empty)
    (on-table green)
    (on red green)
    (clear red)
    (on-table grey)
    (clear grey)
  )
  (:goal
    (and
      (on-table grey)
      (on red grey)
      (on-table green)
    )
  )
)