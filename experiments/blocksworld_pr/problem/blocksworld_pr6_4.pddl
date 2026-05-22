(define (problem blocksworld_pr6_4)
  (:domain blocksworld-original)
  (:objects
    brown cyan green magenta yellow white
  )
  (:init
    (arm-empty)
    (on-table brown)
    (on cyan brown)
    (clear cyan)
    (on-table green)
    (on magenta green)
    (clear magenta)
    (on-table yellow)
    (on white yellow)
    (clear white)
  )
  (:goal
    (and
      (on-table brown)
      (on yellow brown)
      (on magenta yellow)
      (on white magenta)
      (on-table cyan)
      (on green cyan)
    )
  )
)