(define (problem blocksworld_pr4_2)
  (:domain blocksworld-original)
  (:objects
    magenta brown yellow grey
  )
  (:init
    (arm-empty)
    (on-table magenta)
    (on brown magenta)
    (clear brown)
    (on-table yellow)
    (on grey yellow)
    (clear grey)
  )
  (:goal
    (and
      (on-table yellow)
      (on brown yellow)
      (on grey brown)
      (on-table magenta)
    )
  )
)