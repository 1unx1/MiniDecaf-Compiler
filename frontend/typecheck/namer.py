from typing import Protocol, TypeVar, cast

from frontend.ast.node import Node, NullType
from frontend.ast.tree import *
from frontend.ast.visitor import RecursiveVisitor, Visitor
from frontend.scope.globalscope import GlobalScope
from frontend.scope.scope import Scope, ScopeKind
from frontend.scope.scopestack import ScopeStack
from frontend.symbol.funcsymbol import FuncSymbol
from frontend.symbol.symbol import Symbol
from frontend.symbol.varsymbol import VarSymbol
from frontend.type.array import ArrayType
from frontend.type.type import DecafType
from utils.error import *
from utils.riscv import MAX_INT

"""
The namer phase: resolve all symbols defined in the abstract syntax tree and store them in symbol tables (i.e. scopes).
"""


class Namer(Visitor[ScopeStack, None]):
    def __init__(self) -> None:
        pass

    # Entry of this phase
    def transform(self, program: Program) -> Program:
        # Global scope. You don't have to consider it until Step 9.
        program.globalScope = GlobalScope
        ctx = ScopeStack(program.globalScope)

        program.accept(self, ctx)
        return program

    def visitProgram(self, program: Program, ctx: ScopeStack) -> None:
        # Check if the 'main' function is missing
        if not program.hasMainFunc():
            raise DecafNoMainFuncError
        for funcOrDecl in program:
            funcOrDecl.accept(self, ctx)

    def visitFunction(self, func: Function, ctx: ScopeStack) -> None:
        if func.body != NULL: # definination
            declaredFuncSymbol = ctx.findConflict(func.ident.value)
            if declaredFuncSymbol: # declared
                if declaredFuncSymbol.defined: # declared and defined
                    raise DecafDeclConflictError(func.ident.value)
                # declared but not defined
                declaredFuncSymbol.defined = True
                ctx.open(Scope(ScopeKind.LOCAL))
                for param in func.parameter_list:
                    param.accept(self, ctx)
                func.body.func_body = True
                func.body.accept(self, ctx)
                ctx.close()
            else: # not declared, declare and define at the same time
                funcSymbol = FuncSymbol(func.ident.value, func.ret_t.type, ctx.currentScope(), True)
                for param in func.parameter_list:
                    funcSymbol.addParaType(param.var_t)
                ctx.declare(funcSymbol)
                ctx.open(Scope(ScopeKind.LOCAL))
                for param in func.parameter_list:
                    param.accept(self, ctx)
                func.body.func_body = True
                func.body.accept(self, ctx)
                ctx.close()
        else: # declaration
            if ctx.findConflict(func.ident.value):
                raise DecafDeclConflictError(func.ident.value)
            funcSymbol = FuncSymbol(func.ident.value, func.ret_t.type, ctx.currentScope(), False)
            for param in func.parameter_list:
                funcSymbol.addParaType(param.var_t)
            ctx.declare(funcSymbol)

    def visitParameter(self, param: Parameter, ctx: ScopeStack) -> None:
        if ctx.findConflict(param.ident.value):
            raise DecafDeclConflictError(param.ident.value)
        symbol = VarSymbol(param.ident.value, param.var_t.type)
        ctx.declare(symbol)
        param.setattr('symbol', symbol)

    def visitCall(self, call: Call, ctx: ScopeStack) -> None:
        try:
            call.ident.accept(self, ctx)
        except DecafUndefinedVarError:
            raise DecafUndefinedFuncError(call.ident.value)
        funcSymbol = call.ident.getattr('symbol')
        if not isinstance(funcSymbol, FuncSymbol):
            raise DecafUndefinedFuncError(call.ident.value)
        if len(call.argument_list) != funcSymbol.parameterNum:
            raise DecafBadFuncCallError(call.ident.value)
        for argument in call.argument_list:
            argument.accept(self, ctx)

    def visitBlock(self, block: Block, ctx: ScopeStack) -> None:
        if block.func_body:
            for child in block:
                child.accept(self, ctx)
            return
        ctx.open(Scope(ScopeKind.LOCAL))
        for child in block:
            child.accept(self, ctx)
        ctx.close()

    def visitReturn(self, stmt: Return, ctx: ScopeStack) -> None:
        stmt.expr.accept(self, ctx)

    def visitFor(self, stmt: For, ctx: ScopeStack) -> None:
        """
        1. Open a local scope for stmt.init.
        2. Visit stmt.init, stmt.cond, stmt.update.
        3. Open a loop in ctx (for validity checking of break/continue)
        4. Visit body of the loop.
        5. Close the loop and the local scope.
        """
        ctx.open(Scope(ScopeKind.LOCAL))
        if not stmt.init is NULL:
            stmt.init.accept(self, ctx)
        stmt.cond.accept(self, ctx)
        if not stmt.update is NULL:
            stmt.update.accept(self, ctx)
        ctx.openLoop()
        stmt.body.accept(self, ctx)
        ctx.closeLoop()
        ctx.close()

    def visitIf(self, stmt: If, ctx: ScopeStack) -> None:
        stmt.cond.accept(self, ctx)
        stmt.then.accept(self, ctx)

        # check if the else branch exists
        if not stmt.otherwise is NULL:
            stmt.otherwise.accept(self, ctx)

    def visitWhile(self, stmt: While, ctx: ScopeStack) -> None:
        stmt.cond.accept(self, ctx)
        ctx.openLoop()
        stmt.body.accept(self, ctx)
        ctx.closeLoop()
        
    def visitDoWhile(self, stmt: DoWhile, ctx: ScopeStack) -> None:
        """
        1. Open a loop in ctx (for validity checking of break/continue)
        2. Visit body of the loop.
        3. Close the loop.
        4. Visit the condition of the loop.
        """
        ctx.openLoop()
        stmt.body.accept(self, ctx)
        ctx.closeLoop()
        stmt.cond.accept(self, ctx)

    def visitBreak(self, stmt: Break, ctx: ScopeStack) -> None:
        if not ctx.inLoop():
            raise DecafBreakOutsideLoopError()

    def visitContinue(self, stmt: Continue, ctx: ScopeStack) -> None:
        """
        1. Refer to the implementation of visitBreak.
        """
        if not ctx.inLoop():
            raise DecafContinueOutsideLoopError()

    def visitDeclaration(self, decl: Declaration, ctx: ScopeStack) -> None:
        """
        1. Use ctx.findConflict to find if a variable with the same name has been declared.
        2. If not, build a new VarSymbol, and put it into the current scope using ctx.declare.
        3. Set the 'symbol' attribute of decl.
        4. If there is an initial value, visit it.
        """
        isGlobal = ctx.isGlobalScope()
        if ctx.findConflict(decl.ident.value):
            if isGlobal:
                raise DecafGlobalVarDefinedTwiceError(decl.ident.value)
            raise DecafDeclConflictError(decl.ident.value)
        symbol = VarSymbol(decl.ident.value, decl.var_t.type, isGlobal)
        ctx.declare(symbol)
        decl.setattr('symbol', symbol)
        if decl.init_expr != NULL:
            decl.init_expr.accept(self, ctx)
            if isGlobal and not isinstance(decl.init_expr, IntLiteral):
                raise DecafGlobalVarBadInitValueError(decl.ident.value)

    def visitAssignment(self, expr: Assignment, ctx: ScopeStack) -> None:
        """
        1. Refer to the implementation of visitBinary.
        """
        expr.lhs.accept(self, ctx)
        expr.rhs.accept(self, ctx)

    def visitUnary(self, expr: Unary, ctx: ScopeStack) -> None:
        expr.operand.accept(self, ctx)

    def visitBinary(self, expr: Binary, ctx: ScopeStack) -> None:
        expr.lhs.accept(self, ctx)
        expr.rhs.accept(self, ctx)

    def visitCondExpr(self, expr: ConditionExpression, ctx: ScopeStack) -> None:
        """
        1. Refer to the implementation of visitBinary.
        """
        expr.cond.accept(self, ctx)
        expr.then.accept(self, ctx)
        expr.otherwise.accept(self, ctx)

    def visitIdentifier(self, ident: Identifier, ctx: ScopeStack) -> None:
        """
        1. Use ctx.lookup to find the symbol corresponding to ident.
        2. If it has not been declared, raise a DecafUndefinedVarError.
        3. Set the 'symbol' attribute of ident.
        """
        symbol = ctx.lookup(ident.value)
        if not symbol:
            raise DecafUndefinedVarError(ident.value)
        ident.setattr('symbol', symbol)

    def visitIntLiteral(self, expr: IntLiteral, ctx: ScopeStack) -> None:
        value = expr.value
        if value > MAX_INT:
            raise DecafBadIntValueError(value)
