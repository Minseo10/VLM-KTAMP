(define (problem blocksworld_pr5_5)
  (:domain blocksworld-original)
  (:objects
    brown blue magenta red cyan
  )
  (:init
    (arm-empty)
    (on-table brown)
    (on blue brown)
    (clear blue)
    (on-table magenta)
    (on red magenta)
    (clear red)
    (on-table cyan)
    (clear cyan)
  )
  (:goal
    (and
      (on-table red)
      (on cyan red)
      (on-table blue)
      (on magenta blue)
      (on brown magenta)
    )
  )
)