import utils.riscv as riscv
from frontend.ast import node
from frontend.ast.tree import *
from frontend.ast.visitor import Visitor
from frontend.symbol.varsymbol import VarSymbol
from frontend.type.array import ArrayType
from utils.tac import tacop
from utils.tac.funcvisitor import FuncVisitor
from utils.tac.programwriter import ProgramWriter
from utils.tac.tacprog import TACProg
from utils.tac.temp import Temp

"""
The TAC generation phase: translate the abstract syntax tree into three-address code.
"""


class TACGen(Visitor[FuncVisitor, None]):
    def __init__(self) -> None:
        pass

    # Entry of this phase
    def transform(self, program: Program) -> TACProg:
        pw = ProgramWriter([func for func in program.functions()])
        for func_name, func in program.functions().items():
            if func.body != NULL:
                mv = pw.visitFunc(func_name)
                for param in func.parameter_list:
                    temp = mv.freshTemp()
                    param.getattr('symbol').temp = temp
                    mv.func.params.append(temp)
                func.body.accept(self, mv)
                # Remember to call mv.visitEnd after the translation a function.
                mv.visitEnd()
        globalSymbolNameValues = {}
        for decl in program.declarations():
            globalSymbol = decl.getattr('symbol')
            if decl.init_expr != NULL:
                globalSymbol.setInitValue(decl.init_expr.value)
            globalSymbolNameValues[globalSymbol.name] = globalSymbol.initValue
        # Remember to call pw.visitEnd before finishing the translation phase.
        return pw.visitEnd(globalSymbolNameValues)

    def visitCall(self, call: Call, mv: FuncVisitor) -> None:
        for argument in call.argument_list:
            argument.accept(self, mv)
            mv.visitParam(argument.getattr('val'))
        args = [argument.getattr('val') for argument in call.argument_list]
        call.setattr('val', mv.visitCall(mv.ctx.getFuncLabel(call.ident.value), args))

    def addressCompute(self, indexExpr: IndexExpr, mv: FuncVisitor) -> Temp:
        expr = indexExpr
        indexes = []
        expr.index.accept(self, mv)
        indexes.append(expr.index.getattr('val'))
        while not isinstance(expr.base, Identifier):
            expr = expr.base
            expr.index.accept(self, mv)
            indexes.append(expr.index.getattr('val'))
        arraySymbol = expr.base.getattr('symbol')
        if arraySymbol.isGlobal:
            arraySymbol.temp = mv.visitLoadSymbol(arraySymbol.name)
        lengths = []
        type = arraySymbol.type
        while isinstance(type, ArrayType):
            lengths.append(type.length)
            type = type.base
        lengths.append(1)
        lengths.reverse()
        addrTemp = arraySymbol.temp
        size = 4
        for i, index in enumerate(indexes):
            size *= lengths[i]
            addrTemp = mv.visitBinary(tacop.BinaryOp.ADD, addrTemp, mv.visitBinary(
                tacop.BinaryOp.MUL, index, mv.visitLoad(size)
            ))
        return addrTemp

    def visitIndexExpr(self, indexExpr: IndexExpr, mv: FuncVisitor) -> None:
        addrTemp = self.addressCompute(indexExpr, mv)
        indexExpr.setattr('val', (mv.visitLoadInMem(addrTemp, 0)))

    def visitBlock(self, block: Block, mv: FuncVisitor) -> None:
        for child in block:
            child.accept(self, mv)

    def visitReturn(self, stmt: Return, mv: FuncVisitor) -> None:
        stmt.expr.accept(self, mv)
        mv.visitReturn(stmt.expr.getattr("val"))

    def visitBreak(self, stmt: Break, mv: FuncVisitor) -> None:
        mv.visitBranch(mv.getBreakLabel())

    def visitContinue(self, stmt: Continue, mv: FuncVisitor) -> None:
        mv.visitBranch(mv.getContinueLabel())

    def visitIdentifier(self, ident: Identifier, mv: FuncVisitor) -> None:
        """
        1. Set the 'val' attribute of ident as the temp variable of the 'symbol' attribute of ident.
        """
        symbol = ident.getattr('symbol')
        if symbol.isGlobal:
            base = mv.visitLoadSymbol(symbol.name)
            if not isinstance(symbol.type, ArrayType):
                symbol.temp = mv.visitLoadInMem(base, 0)
            else:
                symbol.temp = base
        ident.setattr('val', symbol.temp)

    def visitDeclaration(self, decl: Declaration, mv: FuncVisitor) -> None:
        """
        1. Get the 'symbol' attribute of decl.
        2. Use mv.freshTemp to get a new temp variable for this symbol.
        3. If the declaration has an initial value, use mv.visitAssignment to set it.
        """
        symbol = decl.getattr('symbol')
        if decl.indexes == NULL:
            symbol.temp = mv.freshTemp()
        else:
            symbol.temp = mv.visitAlloc(symbol.type.size)
            offset = 0
            if 0 in symbol.initValue:
                zeroTemp = mv.visitLoad(0)
            for integer in symbol.initValue:
                temp = mv.visitLoad(integer) if integer != 0 else zeroTemp
                mv.visitStoreInMem(temp, symbol.temp, offset)
                offset += 4
            mv.func.arrays.append((symbol.temp, symbol.type.size))
        if decl.init_expr != NULL:
            decl.init_expr.accept(self, mv)
            decl.setattr('val', mv.visitAssignment(symbol.temp, decl.init_expr.getattr('val')))

    def visitAssignment(self, expr: Assignment, mv: FuncVisitor) -> None:
        """
        1. Visit the right hand side of expr, and get the temp variable of left hand side.
        2. Use mv.visitAssignment to emit an assignment instruction.
        3. Set the 'val' attribute of expr as the value of assignment instruction.
        """
        expr.rhs.accept(self, mv)
        if isinstance(expr.lhs, IndexExpr):
            addrTemp = self.addressCompute(expr.lhs, mv)
            mv.visitStoreInMem(expr.rhs.getattr('val'), addrTemp, 0)
            expr.setattr('val', expr.rhs.getattr('val'))
        else:
            symbol = expr.lhs.getattr('symbol')
            if symbol.isGlobal:
                base = mv.visitLoadSymbol(symbol.name)
                mv.visitStoreInMem(expr.rhs.getattr('val'), base, 0)
                expr.setattr('val', expr.rhs.getattr('val'))
            else:
                expr.setattr('val', mv.visitAssignment(symbol.temp, expr.rhs.getattr('val')))

    def visitIf(self, stmt: If, mv: FuncVisitor) -> None:
        stmt.cond.accept(self, mv)

        if stmt.otherwise is NULL:
            skipLabel = mv.freshLabel()
            mv.visitCondBranch(
                tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), skipLabel
            )
            stmt.then.accept(self, mv)
            mv.visitLabel(skipLabel)
        else:
            skipLabel = mv.freshLabel()
            exitLabel = mv.freshLabel()
            mv.visitCondBranch(
                tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), skipLabel
            )
            stmt.then.accept(self, mv)
            mv.visitBranch(exitLabel)
            mv.visitLabel(skipLabel)
            stmt.otherwise.accept(self, mv)
            mv.visitLabel(exitLabel)

    def visitWhile(self, stmt: While, mv: FuncVisitor) -> None:
        beginLabel = mv.freshLabel()
        loopLabel = mv.freshLabel()
        breakLabel = mv.freshLabel()
        mv.openLoop(breakLabel, loopLabel)

        mv.visitLabel(beginLabel)
        stmt.cond.accept(self, mv)
        mv.visitCondBranch(tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), breakLabel)

        stmt.body.accept(self, mv)
        mv.visitLabel(loopLabel)
        mv.visitBranch(beginLabel)
        mv.visitLabel(breakLabel)
        mv.closeLoop()

    def visitFor(self, stmt: For, mv: FuncVisitor) -> None:
        beginLabel = mv.freshLabel()
        loopLabel = mv.freshLabel()
        breakLabel = mv.freshLabel()
        mv.openLoop(breakLabel, loopLabel)
        if not stmt.init is NULL:
            stmt.init.accept(self, mv)
        mv.visitLabel(beginLabel)
        stmt.cond.accept(self, mv)
        mv.visitCondBranch(tacop.CondBranchOp.BEQ, stmt.cond.getattr('val'), breakLabel)
        stmt.body.accept(self, mv)
        mv.visitLabel(loopLabel)
        if not stmt.update is NULL:
            stmt.update.accept(self, mv)
        mv.visitBranch(beginLabel)
        mv.visitLabel(breakLabel)
        mv.closeLoop()

    def visitDoWhile(self, stmt: DoWhile, mv: FuncVisitor) -> None:
        beginLabel = mv.freshLabel()
        loopLabel = mv.freshLabel()
        breakLabel = mv.freshLabel()
        mv.openLoop(breakLabel, loopLabel)
        mv.visitLabel(beginLabel)
        stmt.body.accept(self, mv)
        mv.visitLabel(loopLabel)
        stmt.cond.accept(self, mv)
        mv.visitCondBranch(tacop.CondBranchOp.BEQ, stmt.cond.getattr('val'), breakLabel)
        mv.visitBranch(beginLabel)
        mv.visitLabel(breakLabel)
        mv.closeLoop()

    def visitUnary(self, expr: Unary, mv: FuncVisitor) -> None:
        expr.operand.accept(self, mv)

        op = {
            node.UnaryOp.Neg: tacop.UnaryOp.NEG,
            node.UnaryOp.BitNot: tacop.UnaryOp.NOT,
            node.UnaryOp.LogicNot: tacop.UnaryOp.SEQZ,
            # You can add unary operations here.
        }[expr.op]
        expr.setattr("val", mv.visitUnary(op, expr.operand.getattr("val")))

    def visitBinary(self, expr: Binary, mv: FuncVisitor) -> None:
        expr.lhs.accept(self, mv)
        expr.rhs.accept(self, mv)

        op = {
            node.BinaryOp.Add: tacop.BinaryOp.ADD,
            node.BinaryOp.Sub: tacop.BinaryOp.SUB,
            node.BinaryOp.Mul: tacop.BinaryOp.MUL,
            node.BinaryOp.Div: tacop.BinaryOp.DIV,
            node.BinaryOp.Mod: tacop.BinaryOp.REM,
            node.BinaryOp.EQ: tacop.BinaryOp.EQU,
            node.BinaryOp.NE: tacop.BinaryOp.NEQ,
            node.BinaryOp.LT: tacop.BinaryOp.SLT,
            node.BinaryOp.GT: tacop.BinaryOp.SGT,
            node.BinaryOp.LE: tacop.BinaryOp.LEQ,
            node.BinaryOp.GE: tacop.BinaryOp.GEQ,
            node.BinaryOp.LogicAnd: tacop.BinaryOp.AND,
            node.BinaryOp.LogicOr: tacop.BinaryOp.OR,
            # You can add binary operations here.
        }[expr.op]
        expr.setattr(
            "val", mv.visitBinary(op, expr.lhs.getattr("val"), expr.rhs.getattr("val"))
        )

    def visitCondExpr(self, expr: ConditionExpression, mv: FuncVisitor) -> None:
        """
        1. Refer to the implementation of visitIf and visitBinary.
        """
        expr.cond.accept(self, mv)
        skipLabel = mv.freshLabel()
        exitLabel = mv.freshLabel()
        temp = mv.freshTemp()
        mv.visitCondBranch(tacop.CondBranchOp.BEQ, expr.cond.getattr('val'), skipLabel)
        expr.then.accept(self, mv)
        mv.visitAssignment(temp, expr.then.getattr('val'))
        mv.visitBranch(exitLabel)
        mv.visitLabel(skipLabel)
        expr.otherwise.accept(self, mv)
        mv.visitAssignment(temp, expr.otherwise.getattr('val'))
        mv.visitLabel(exitLabel)
        expr.setattr('val', temp)

    def visitIntLiteral(self, expr: IntLiteral, mv: FuncVisitor) -> None:
        expr.setattr("val", mv.visitLoad(expr.value))
