(define (problem blocksworld_pr5_2)
  (:domain blocksworld-original)
  (:objects
    cyan blue magenta green grey
  )
  (:init
    (arm-empty)
    (on-table cyan)
    (on blue cyan)
    (on magenta blue)
    (clear magenta)
    (on-table green)
    (on grey green)
    (clear grey)
  )
  (:goal
    (and
      (on-table grey)
      (on-table cyan)
      (on green cyan)
      (on magenta green)
      (on blue magenta)
    )
  )
)