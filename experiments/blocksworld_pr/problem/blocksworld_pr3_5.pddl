(define (problem prob)
  (:domain blocksworld-original)
  (:objects
    red green blue
  )
  (:init
    (arm-empty)
    (on-table blue)
    (on green blue)
    (on red green)
    (clear red)
  )
  (:goal
    (and
    (on-table red)
    (on green red)
    (on blue green)
     )))
