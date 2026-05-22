(define (problem blocksworld_pr6_3)
  (:domain blocksworld-original)
  (:objects
    green grey red brown cyan magenta
  )
  (:init
    (arm-empty)
    (on-table green)
    (on grey green)
    (on red grey)
    (clear red)
    (on-table brown)
    (on cyan brown)
    (on magenta cyan)
    (clear magenta)
  )
  (:goal
    (and
      (on-table grey)
      (on brown grey)
      (on cyan brown)
      (on-table magenta)
      (on red magenta)
      (on green red)
    )
  )
)