#!/usr/bin/env python3
import re
import sys
import signal
import argparse
from collections import defaultdict
from copy import copy,deepcopy
from itertools import product
from nltk.sem import Expression
from nltk.inference import mace, prover9

class Formula:
    
    def __init__(self, ftuple, 
                 predicates=None, freeVariables=None, 
                 subformulas=None):
        self.ftuple = ftuple
        self.mainOperator = ftuple[0]
        self.predicates = predicates # arity => predicates in alphabetic order
        self.freeVariables = freeVariables # in alphabetic order
        self.subformulas = subformulas
        self.disjuncts = ((ftuple[0] in ('exists', '|', '->') and subformulas)
                          or (ftuple[0] == '-' and subformulas[0].conjuncts)
                          or [])
        self.conjuncts = ((ftuple[0] in ('all', '&') and subformulas)
                          or (ftuple[0] == '-' and subformulas[0].disjuncts)
                          or [])

    def __eq__(self, other):
        return self.ftuple == other.ftuple

    def negated(self):
        return Formula(('-',) + self.ftuple,
                    predicates=self.predicates,
                    freeVariables=self.freeVariables,
                    subformulas=(self,))
    
    def quantified(self, quantifier, variable):
        freevars = copy(self.freeVariables)
        freevars.remove(variable)
        return Formula((quantifier,variable) + self.ftuple,
                    predicates=self.predicates,
                    freeVariables=freevars,
                    subformulas=(self,))
    
    def conjoined(self, symbol, f2):
        predicates = self.merge(self.predicates, f2.predicates)
        freevars = self.merge(self.freeVariables, f2.freeVariables)
        return Formula((symbol,) + self.ftuple + f2.ftuple,
                       predicates=predicates,
                       freeVariables=freevars,
                       subformulas=(self, f2))
    
    def vocabulary(self, merge=None):
        if merge:
            return {
                'variables': self.merge(self.freeVariables, merge['variables']),
                'predicates': self.merge(self.predicates, merge['predicates'])
            }
        else:
            return {
                'variables': self.freeVariables,
                'predicates': self.predicates
            }

    @staticmethod
    def merge(col1, col2):
        if isinstance(col1, list): # variable lists
            return sorted(list(set(col1 + col2)))
        else: # predicate dicts
            ps = copy(col1) 
            for arity, preds in col2.items():
                if arity not in ps:
                    ps[arity] = preds
                else:
                    ps[arity] = sorted(list(set(ps[arity] + preds)))
            return ps

    def signature(self):
        if not hasattr(self, '_sig'):
            self._sig = ''
            mapping = {}  
            var_index, pred_index = 0, 0
            for symbol in self.ftuple:
                if symbol in ('-','&','|','->','<->','all','exists'):
                    self._sig += symbol
                else:
                    if symbol not in mapping:
                        if self.isPredicate(symbol):
                            pred_index += 1
                            mapping[symbol] = 'F'+str(pred_index)
                        elif self.isVariable(symbol):
                            var_index += 1
                            mapping[symbol] = 'x'+str(var_index)
                    self._sig += mapping[symbol]
        return self._sig
    
    def __repr__(self, bound_vars=None):
        # replace free variables by constant symbols:
        if not bound_vars:
            bound_vars = []
        if self.isPredicate(self.ftuple[0]):
            terms = ['a'+t[1:] if t not in bound_vars else t for t in self.ftuple[1:]]
            return '{}({})'.format(self.ftuple[0], ','.join(terms))
        elif self.ftuple[0] == '-':
            return '-{}'.format(self.subformulas[0].__repr__(bound_vars=bound_vars))
        elif self.ftuple[0] in ('all', 'exists'):
            bound_vars = bound_vars + [self.ftuple[1]]
            return '{} {}.{}'.format(self.ftuple[0], self.ftuple[1], 
                                     self.subformulas[0].__repr__(bound_vars=bound_vars))
        else:
            return '({} {} {})'.format(self.subformulas[0].__repr__(bound_vars=bound_vars), 
                                       self.ftuple[0], 
                                       self.subformulas[1].__repr__(bound_vars=bound_vars))
        
    @staticmethod
    def isPredicate(symbol):
        return symbol[0] == 'F'
 
    @staticmethod
    def isVariable(symbol):
        return symbol[0] == 'x'

class AtomicFormula(Formula):

    def __init__(self, predicate, terms, **kwargs):
        ftuple = (predicate,) + tuple(terms)
        super().__init__(ftuple,
                         predicates={len(terms): [predicate]},
                         freeVariables=sorted(list(set(terms))),
                         **kwargs)

# ==============================================================================

MAX_PREDICATES = float('inf')
MAX_VARIABLES = float('inf')
LENGTH = 2

def formulas():
    # skip existential or disjunctive formulas, and formulas that are
    # conjuncts of disjunctive formulas etc.:
    def disjunctive(f):
        if len(f.disjuncts) > 1:
            return True
        for subf in f.conjuncts:
            if disjunctive(subf):
                return True
        return False
    exclude = ('|', '->', 'exists')
    for f in subformulas(LENGTH, exclude=exclude):
        if not disjunctive(f):
            yield f

def subformulas(length, lvocab=None, exclude=None):
    if length <= 1:
        return
    exclude = exclude or []

    for f in quantifiedFormulas(length, lvocab=lvocab, exclude=exclude):
        yield f
    if '-' not in exclude:
        for f in negatedFormulas(length, lvocab=lvocab):
            yield f
    for f in conjoinedFormulas(length, lvocab=lvocab, exclude=exclude):
        yield f
    # don't check silly things like F(a,b,c,d,e) or F(a,b,c) -> G(a,b,c):
    if length < max(4, LENGTH/3):
        for f in atomicFormulas(length, lvocab=lvocab):
            yield f

def skip_or(f1, f2):
    return (f1.mainOperator == '-' # skip -AvB, since A->B is shorter
            or f2.mainOperator == '-' # Av-B => B->A
            or f1.mainOperator == '&' and f2 in f1.subformulas # (A&B)vA => A 
            or f1 == f2) # AvA => A 

def skip_and(f1, f2):
    return (f1 == f2 # A&A => A
            or f1.mainOperator == '-' == f2.mainOperator # -A&-B => -(AvB)
            or (f1.mainOperator == '-' and f1.subformulas[0] == f2) # -A&A => 0
            or (f2.mainOperator == '-' and f2.subformulas[0] == f1) # A&-A => 0 
            or (f1.mainOperator == 'v' and f2 in f1.subformulas)) # (AvB)&A => A 

def skip_cond(f1, f2):
    return (f1.mainOperator == '-' # -A->B => AvB
            or f2.mainOperator in ('-','->') # A->-B => -(A&B), A->(B->C) => (A&B)->C
            or f1 == f2) # A->A => 1 

def skip_bicond(f1, f2):
    return (f2.mainOperator == '-' # A<->-B => -A<->B
            or f1 == f2) # A<->A => 1

def skip_not(f):
    # f.mainOperator can only be '&' or '|' or 'Fi' to begin with
    if not f.mainOperator in ('&','|'):
        return False
    for sub in f.subformulas:
        if (sub.mainOperator == '-' # -(A&-B) => -AvB, -(A|-B) => -A&B
            or (sub.mainOperator in ('all', 'exists') 
                and sub.subformulas[0].mainOperator == '-')):
                # -(all x -A & B) => exists x A & -B
            return True
    return False

skip = {
    '|': skip_or,
    '&': skip_and,
    '->': skip_cond,
    '<->': skip_bicond,
    '-': skip_not
}

def negatedFormulas(length, lvocab=None):
    for f in subformulas(length-1, lvocab=lvocab, exclude=('-','->','<->','all','exists')):
        if not skip['-'](f):
            yield f.negated()
    
def quantifiedFormulas(length, lvocab=None, exclude=None):
    quantifiers = [q for q in ('all', 'exists') if q not in exclude]
    if not quantifiers:
        return
    for f in subformulas(length-2, lvocab=lvocab):
        # no point trying both Ax.Fxy and Ay.Fxy if F isn't yet used
        # anywhere else:
        variables = f.freeVariables if lvocab else f.freeVariables[:1]
        for v in variables:
            for q in quantifiers:
                yield f.quantified(q, v)

def conjoinedFormulas(length, lvocab=None, exclude=None):
    operators = [o for o in ('->', '&', '|', '<->') if o not in exclude]
    # &,v,<-> are commutative, so here we enforce the LHS to be at
    # least as long as the RHS:
    MIN_FORMULA_LENGTH = 2
    min_splitpoint = MIN_FORMULA_LENGTH if '->' in operators else int(length/2)
    for splitpoint in range((length-1)-MIN_FORMULA_LENGTH, min_splitpoint-1, -1):
        #yielded = []
        for f1 in subformulas(splitpoint, lvocab=lvocab):
            llvocab = f1.vocabulary(merge=lvocab)
            for f2 in subformulas((length-splitpoint)-1, lvocab=llvocab):
                # skip new formulas that would be obvious tautologies,
                # contradictions or equivalent to one side:
                for o in operators:
                    # enforce the LHS to be at least as long as the
                    # RHS for '<->','&','|':
                    if o != '->' and splitpoint < int(length/2):
                        continue
                    if skip[o](f1, f2):
                        continue
                    conj = f1.conjoined(o, f2)
                    #if not any(alpha_variant(conj, y) for y in yielded):
                    #    yielded.append(conj)
                    yield(conj)
                        
def atomicFormulas(length, lvocab=None):
    if length < 2:
        return
    arity = length-1
    if lvocab and arity in lvocab['predicates']:
        predicates = lvocab['predicates'][arity] + next_predicates(arity, lvocab)
    else:
        predicates = next_predicates(arity, lvocab)
    lvariables = lvocab['variables'] if lvocab else []
    nvariables = next_variables(arity, lvocab)
    # order/identity doesn't matter for new terms but does for old ones
    for predicate in predicates:
        for lterms in product(lvariables + [' '], repeat=arity):
            # all permutations (with repetitions) of the old variables
            # as well as placeholder spaces for new variables; now
            # fill in the spaces:
            for nterms in newtermlists(nvariables, lterms.count(' ')):
                terms = list(lterms) # also copy
                for t in nterms:
                    terms[terms.index(' ')] = t
                yield AtomicFormula(predicate, terms)

def next_predicates(arity, lvocab):
    if lvocab:
        all_preds = set().union(*lvocab['predicates'].values())
        next_index = len(all_preds)+1
    else:
        next_index = 1
    if next_index <= MAX_PREDICATES:
        return ['F'+str(next_index)]
    else:
        return []

def next_variables(arity, lvocab):
    if lvocab and lvocab['variables']:
        next_index = int(lvocab['variables'][-1][1:]) + 1
    else:
        next_index = 1
    return ['x'+str(next_index+i) for i in range(arity) if next_index+i <= MAX_VARIABLES]

def permutations_with_repetition(iterable, length, incl_alpha_vars=True):
    # ('ABC', 2, True) => AA AB AC BA BB BC CA CB CC
    # ('ABC', 2, False) => AA AB
    # ('ABCD', 3, False) => AAA AAB ABA ABB ABC
    if incl_alpha_vars:
        for perm in product(iterable, repeat=length):
            yield perm
    else:
        pool = tuple(iterable)
        n = len(pool)
        indices = [0] * length
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(length)):
                if indices[i] != n-1 and indices[i] in indices[0:i]:
                    break
            else:
                return
            indices[i] += 1
            if i != length-1:
                indices[i+1:] = [0] * ((length-1)-i)
            yield tuple(pool[i] for i in indices)

def newtermlists(iterable, length):
    # like permutations_with_repetition(iterable, length, False), but
    # also insensitive to order, i.e. 'ABCD' => AAA AAB ABC
    pool = tuple(iterable)
    n = len(pool)
    if not n and length:
        return
    indices = [0] * length
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(length)):
            if indices[i] != n-1 and indices[i] == indices[i-1]:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, length):
            indices[j] = indices[i]
        for j in range(1,n):
            if indices.count(j) > indices.count(j-1):
                break
        else:
            yield tuple(pool[i] for i in indices)
        
def alpha_variant(formula1, formula2):
    """
    checks if formula1 and formula2 differ only by choice of term or
    predicate symbols
    """
    return formula1.signature() == formula2.signature()
    list1 = [el for el in re.split(r'([Fax]\d+)', formula1) if el]
    list2 = [el for el in re.split(r'([Fax]\d+)', formula2) if el]
    # now lists ares something like ['(', 'F1', '(', 'x12', ') -> (', ...]
    mapping = {}
    rev_mapping = {}
    for el1, el2 in zip(list1, list2):
        for ch in ('F', 'x', 'a'):
            if el1[0] == ch:
                if el2[0] != ch:
                    return False
                if el1 in mapping and mapping[el1] != el2:
                    return False
                mapping[el1] = el2
                if el2 in rev_mapping and rev_mapping[el2] != el1:
                    return False
                rev_mapping[el2] = el1
                break
        else:
            if el1 != el2:
                return False
    return True

# ==============================================================================

def has_singleton_model(formula):

    predicates = [p for plist in formula.predicates.values() for p in plist]

    def evaluate(f):
        ftype = f.mainOperator
        if ftype == '-':
            return not evaluate(f.subformulas[0])
        if ftype in ('all', 'exists'):
            return evaluate(f.subformulas[0])
        elif ftype == '&':
            return evaluate(f.subformulas[0]) and evaluate(f.subformulas[1])
        elif ftype == '|':
            return evaluate(f.subformulas[0]) or evaluate(f.subformulas[1])
        elif ftype == '->':
            return not evaluate(f.subformulas[0]) or evaluate(f.subformulas[1])
        elif ftype == '<->':
            return evaluate(f.subformulas[0]) == evaluate(f.subformulas[1])
        else:
            return interp[predicates.index(ftype)]

    for interp in product((0,1), repeat=len(predicates)):
        if evaluate(formula):
            return True
    return False

expr = Expression.fromstring

class Timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def provable(expression, timeout=1):
    try:
        prover = provable._provers[timeout]
    except Exception:
        if not hasattr(provable, 'provers'):
            provable._provers = {}
        if timeout not in provable._provers:
            provable._provers[timeout] = prover9.Prover9(timeout=timeout)
        prover = provable._provers[timeout]
    return prover.prove(expression)

def model_size(expression, max_models=0):
    mc = mace.MaceCommand(None, [expression], max_models=max_models)
    # NB: mace models always have size at least 2!
    if mc.build_model():
        descr = mc.model(format='cooked')
        size = descr.split('of size ',1)[1].split('\n',1)[0]
        return int(size)
    else:
        return None

def check_formula(formula):
    #print('checking {}'.format(formula))
    if has_singleton_model(formula):
        return 1
    e = expr(formula.__repr__())
    with Timeout(seconds=20):
        for timeout in range(1,10):
            size = model_size(e, max_models=timeout*10)
            if size:
                return size
            ne = e.negate()
            if provable(ne, timeout=timeout):
                return None
    print('done')
    raise TimeoutError

# ==============================================================================

def main():
    greatest_size, greatest_formulas = 1, []
    prover_errors = []
    for i,f in enumerate(formulas()):
        fstr = f.__repr__()
        try:
            size = check_formula(f)
        except Exception as e:
            print('***** prover crashed while processing {}: {}'.format(fstr, e))
            prover_errors.append(fstr)
            continue
        if size:
            if size >= greatest_size and has_singleton_model(f):
                size = 1
            elif size > greatest_size:
                greatest_size, greatest_formulas = size, []
            if size >= greatest_size: 
                greatest_formulas.append(fstr)
            print('{}: {}\tmodel size: {} (record {})'.format(i, f, size, greatest_size))
        else:
            print('{}: {}\tformula has no model'.format(i, f))
    print('greatest size: {}'.format(greatest_size))
    print('formulas:\n  {}'.format('\n  '.join(greatest_formulas)))
    print('errors:\n  {}'.format('\n  '.join(prover_errors)))

ap = argparse.ArgumentParser()
ap.add_argument('length', type=int, help='length of formulas')
ap.add_argument('--maxpredicates', default=None, type=int, help='maximum number of different predicates')
ap.add_argument('--maxvariables', default=None, type=int, help='maximum number of different variables')
args = ap.parse_args()

LENGTH = args.length
if args.maxpredicates:
    MAX_PREDICATES = args.maxpredicates
if args.maxvariables:
    MAX_VARIABLES = args.maxvariables

main()

# prover9 often crashes, e.g.:
# prover = prover9.Prover9(timeout=10)
# prover.prove(expr('((((F(a) -> F(b)) -> F(c)) <-> F(c)) <-> F(a))'))

# mace4 hangs without even crashing:
#e = expr('all x2.(all x2.exists x1.-F1(x1,x2) <-> all x1.F1(x1,x2))')
#mc = mace.MaceCommand(None, [e], timeout=5)
#mc.build_model()

