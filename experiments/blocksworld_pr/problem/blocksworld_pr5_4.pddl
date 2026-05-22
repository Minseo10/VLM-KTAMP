(define (problem blocksworld_pr5_4)
  (:domain blocksworld-original)
  (:objects
    magenta red grey yellow cyan
  )
  (:init
    (arm-empty)
    (on-table magenta)
    (on red magenta)
    (on grey red)
    (clear grey)
    (on-table yellow)
    (on cyan yellow)
    (clear cyan)
  )
  (:goal
    (and
      (on-table grey)
      (on-table yellow)
      (on magenta yellow)
      (on cyan magenta)
      (on red cyan)
    )
  )
)