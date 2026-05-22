(define (domain blocksworld-original)
  (:requirements :strips)
(:predicates (clear ?x) ; no object is on top of ?x
             (on-table ?x) ; ?x is directly on the table
             (arm-empty) ; the left arm is empty (shown as the right arm in the picture)
             (holding ?x) ; the arm is holding ?x
             (on ?x ?y) ; ?x is stacked on top of ?y
)
; pick up ?ob from the table into the arm
(:action pickup
  :parameters (?ob)
  :precondition (and (clear ?ob) (on-table ?ob) (arm-empty))
  :effect (and (holding ?ob) (not (clear ?ob)) (not (on-table ?ob))
               (not (arm-empty))))
; put down ?ob from the arm onto the table
(:action putdown
  :parameters  (?ob)
  :precondition (holding ?ob)
  :effect (and (clear ?ob) (arm-empty) (on-table ?ob)
               (not (holding ?ob))))
; stack ?ob (held) onto ?underob (which must be clear)
(:action stack
  :parameters  (?ob ?underob)
  :precondition (and (clear ?underob) (holding ?ob))
  :effect (and (arm-empty) (clear ?ob) (on ?ob ?underob)
               (not (clear ?underob)) (not (holding ?ob))))
; unstack ?ob from on top of ?underob into the arm
(:action unstack
  :parameters  (?ob ?underob)
  :precondition (and (on ?ob ?underob) (clear ?ob) (arm-empty))
  :effect (and (holding ?ob) (clear ?underob)
               (not (on ?ob ?underob)) (not (clear ?ob)) (not (arm-empty)))))