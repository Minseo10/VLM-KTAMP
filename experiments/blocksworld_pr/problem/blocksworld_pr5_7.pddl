(define (problem prob)
  (:domain pr2-blocksworld)
  (:objects
    red green blue yellow grey
  )
  (:init
    (arm-empty)
    (on-table blue)
    (on-table red)
    (on-table green)
    (on-table yellow)
    (on-table grey)
    (clear red)
    (clear green)
    (clear blue)
    (clear yellow)
    (clear grey)
  )
  (:goal
    (and
    (on-table red)
    (on green red)
    (on blue green)
    (on yellow blue)
    (on grey yellow)
     )))
