(define (problem blocksworld_pr3_1)
  (:domain blocksworld-original)
  (:objects
    green red blue
  )
  (:init
    (arm-empty)
    (on-table green)
    (on red green)
    (clear red)
    (on-table blue)
    (clear blue)
  )
  (:goal
    (and
      (on-table blue)
      (on red blue)
      (on-table green)
    )
  )
)