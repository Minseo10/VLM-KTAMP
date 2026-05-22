(define (problem blocksworld_pr6_2)
  (:domain blocksworld-original)
  (:objects
    magenta blue cyan brown grey white
  )
  (:init
    (arm-empty)
    (on-table magenta)
    (on blue magenta)
    (on cyan blue)
    (clear cyan)
    (on-table brown)
    (on grey brown)
    (on white grey)
    (clear white)
  )
  (:goal
    (and
      (on-table blue)
      (on magenta blue)
      (on-table cyan)
      (on white cyan)
      (on grey white)
      (on brown grey)
    )
  )
)