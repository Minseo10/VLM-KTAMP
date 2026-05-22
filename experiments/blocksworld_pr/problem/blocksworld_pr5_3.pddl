(define (problem blocksworld_pr5_3)
  (:domain blocksworld-original)
  (:objects
    green red yellow cyan magenta
  )
  (:init
    (arm-empty)
    (on-table green)
    (on red green)
    (on yellow red)
    (clear yellow)
    (on-table cyan)
    (on magenta cyan)
    (clear magenta)
  )
  (:goal
    (and
      (on-table green)
      (on yellow green)
      (on-table red)
      (on cyan red)
      (on magenta cyan)
    )
  )
)