(define (problem blocksworld_pr4_1)
  (:domain blocksworld-original)
  (:objects
    green grey yellow blue
  )
  (:init
    (arm-empty)
    (on-table green)
    (on grey green)
    (clear grey)
    (on-table yellow)
    (on blue yellow)
    (clear blue)
  )
  (:goal
    (and
      (on-table grey)
      (on blue grey)
      (on-table yellow)
      (on green yellow)
    )
  )
)