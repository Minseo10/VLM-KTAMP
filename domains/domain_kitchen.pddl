(define (domain kitchen)
  (:requirements :strips :typing :equality :negative-preconditions)
  (:types
    object
    phys - object
    sink stove table - phys
  )
  (:constants
    mytable - table
    mystove - stove
    mysink  - sink
    egg bacon celery radish chicken apple - object
  )
  (:predicates
    (hand-empty)
    (flow-ok)
    (must-clean ?obj - object)
    (must-cook  ?obj - object)
    (holding ?obj - object)
    (cleaned ?obj - object)
    (cooked  ?obj - object)
    (on ?obj - object ?on - phys)
  )

  (:action pickup
    :parameters (?obj - object ?on - phys)
    :precondition (and
      (on ?obj ?on)
      (hand-empty)
      (flow-ok)
    )
    :effect (and
      (holding ?obj)
      (not (on ?obj ?on))
      (not (hand-empty))
    )
  )

  ;; putdown on sink
  (:action putdown_sink
    :parameters (?obj - object ?s - sink)
    :precondition (and
      (holding ?obj)
      (flow-ok)
    )
    :effect (and
      (not (holding ?obj))
      (on ?obj ?s)
      (hand-empty)
      (must-clean ?obj)
      (not (flow-ok))
    )
  )

  ;; putdown on stove
  (:action putdown_stove
    :parameters (?obj - object ?st - stove)
    :precondition (and
      (holding ?obj)
      (flow-ok)
    )
    :effect (and
      (not (holding ?obj))
      (on ?obj ?st)
      (hand-empty)
      (must-cook ?obj)
      (not (flow-ok))
    )
  )

  ;; putdown on table (no extra side effects)
  (:action putdown_table
    :parameters (?obj - object ?t - table)
    :precondition (and
      (holding ?obj)
      (flow-ok)
    )
    :effect (and
      (not (holding ?obj))
      (on ?obj ?t)
      (hand-empty)
    )
  )

  (:action clean
    :parameters (?obj - object ?s - sink)
    :precondition (and
      (on ?obj ?s)
      (must-clean ?obj)
    )
    :effect (and
      (cleaned ?obj)
      (not (must-clean ?obj))
      (flow-ok)
    )
  )

  ; object need to be cooked on the stove after cooking
  (:action cook
    :parameters (?obj - object ?st - stove)
    :precondition (and
      (on ?obj ?st)
      (cleaned ?obj)
      (must-cook ?obj)
    )
    :effect (and
      (cooked ?obj)
      (not (cleaned ?obj))
      (not (must-cook ?obj))
      (flow-ok)
    )
  )
)
