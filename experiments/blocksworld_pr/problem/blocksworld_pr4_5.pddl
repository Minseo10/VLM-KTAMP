(define (problem blocksworld_pr4_5)
  (:domain blocksworld-original)
  (:objects
    yellow brown grey magenta
  )
  (:init
    (arm-empty)
    (on-table yellow)
    (on brown yellow)
    (on grey brown)
    (on magenta grey)
    (clear magenta)
  )
  (:goal
    (and
      (on-table brown)
      (on magenta brown)
      (on grey magenta)
      (on yellow grey)
    )
  )
)