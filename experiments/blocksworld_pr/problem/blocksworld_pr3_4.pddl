(define (problem blocksworld_pr3_4)
  (:domain blocksworld-original)
  (:objects
    cyan brown yellow
  )
  (:init
    (arm-empty)
    (on-table cyan)
    (on brown cyan)
    (clear brown)
    (on-table yellow)
    (clear yellow)
  )
  (:goal
    (and
      (on-table brown)
      (on yellow brown)
      (on cyan yellow)
    )
  )
)