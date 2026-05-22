(define (problem blocksworld_pr4_3)
  (:domain blocksworld-original)
  (:objects
    brown green yellow cyan
  )
  (:init
    (arm-empty)
    (on-table brown)
    (on green brown)
    (clear green)
    (on-table yellow)
    (on cyan yellow)
    (clear cyan)
  )
  (:goal
    (and
      (on-table brown)
      (on green brown)
      (on yellow green)
      (on cyan yellow)
    )
  )
)