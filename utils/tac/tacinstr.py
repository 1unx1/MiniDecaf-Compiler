from enum import Enum, auto, unique
from typing import Any, Optional, Union

from utils.label.label import Label
from utils.tac.nativeinstr import NativeInstr
from utils.tac.reg import Reg

from .tacop import *
from .tacvisitor import TACVisitor
from .temp import Temp


class TACInstr:
    def __init__(
        self,
        kind: InstrKind,
        dsts: list[Temp],
        srcs: list[Temp],
        label: Optional[Label],
    ) -> None:
        self.kind = kind
        self.dsts = dsts.copy()
        self.srcs = srcs.copy()
        self.label = label

    def getRead(self) -> list[int]:
        return [src.index for src in self.srcs]

    def getWritten(self) -> list[int]:
        return [dst.index for dst in self.dsts]

    def isLabel(self) -> bool:
        return self.kind is InstrKind.LABEL

    def isSequential(self) -> bool:
        return self.kind == InstrKind.SEQ

    def isReturn(self) -> bool:
        return self.kind == InstrKind.RET

    def toNative(self, dstRegs: list[Reg], srcRegs: list[Reg]) -> NativeInstr:
        oldDsts = dstRegs
        oldSrcs = srcRegs
        self.dsts = dstRegs
        self.srcs = srcRegs
        instrString = self.__str__()
        newInstr = NativeInstr(self.kind, dstRegs, srcRegs, self.label, instrString)
        self.dsts = oldDsts
        self.srcs = oldSrcs
        return newInstr

    def accept(self, v: TACVisitor) -> None:
        pass


# Assignment instruction.
class Assign(TACInstr):
    def __init__(self, dst: Temp, src: Temp) -> None:
        super().__init__(InstrKind.SEQ, [dst], [src], None)
        self.dst = dst
        self.src = src

    def __str__(self) -> str:
        return "%s = %s" % (self.dst, self.src)

    def accept(self, v: TACVisitor) -> None:
        v.visitAssign(self)


# Loading an immediate 32-bit constant.
class LoadImm4(TACInstr):
    def __init__(self, dst: Temp, value: int) -> None:
        super().__init__(InstrKind.SEQ, [dst], [], None)
        self.dst = dst
        self.value = value

    def __str__(self) -> str:
        return "%s = %d" % (self.dst, self.value)

    def accept(self, v: TACVisitor) -> None:
        v.visitLoadImm4(self)


# Unary operations.
class Unary(TACInstr):
    def __init__(self, op: UnaryOp, dst: Temp, operand: Temp) -> None:
        super().__init__(InstrKind.SEQ, [dst], [operand], None)
        self.op = op
        self.dst = dst
        self.operand = operand

    def __str__(self) -> str:
        return "%s = %s %s" % (
            self.dst,
            ("-" if (self.op == UnaryOp.NEG) else "!"),
            self.operand,
        )

    def accept(self, v: TACVisitor) -> None:
        v.visitUnary(self)


# Binary Operations.
class Binary(TACInstr):
    def __init__(self, op: BinaryOp, dst: Temp, lhs: Temp, rhs: Temp) -> None:
        super().__init__(InstrKind.SEQ, [dst], [lhs, rhs], None)
        self.op = op
        self.dst = dst
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        opStr = {
            BinaryOp.ADD: "+",
            BinaryOp.SUB: "-",
            BinaryOp.MUL: "*",
            BinaryOp.DIV: "/",
            BinaryOp.REM: "%",
            BinaryOp.EQU: "==",
            BinaryOp.NEQ: "!=",
            BinaryOp.SLT: "<",
            BinaryOp.LEQ: "<=",
            BinaryOp.SGT: ">",
            BinaryOp.GEQ: ">=",
            BinaryOp.AND: "&&",
            BinaryOp.OR: "||",
        }[self.op]
        return "%s = (%s %s %s)" % (self.dst, self.lhs, opStr, self.rhs)

    def accept(self, v: TACVisitor) -> None:
        v.visitBinary(self)


# Branching instruction.
class Branch(TACInstr):
    def __init__(self, target: Label) -> None:
        super().__init__(InstrKind.JMP, [], [], target)
        self.target = target

    def __str__(self) -> str:
        return "branch %s" % str(self.target)

    def accept(self, v: TACVisitor) -> None:
        v.visitBranch(self)


# Branching with conditions.
class CondBranch(TACInstr):
    def __init__(self, op: CondBranchOp, cond: Temp, target: Label) -> None:
        super().__init__(InstrKind.COND_JMP, [], [cond], target)
        self.op = op
        self.cond = cond
        self.target = target

    def __str__(self) -> str:
        return "if (%s %s) branch %s" % (
            self.cond,
            "== 0" if self.op == CondBranchOp.BEQ else "!= 0",
            str(self.target),
        )

    def accept(self, v: TACVisitor) -> None:
        v.visitCondBranch(self)


# Return instruction.
class Return(TACInstr):
    def __init__(self, value: Optional[Temp]) -> None:
        if value is None:
            super().__init__(InstrKind.RET, [], [], None)
        else:
            super().__init__(InstrKind.RET, [], [value], None)
        self.value = value

    def __str__(self) -> str:
        return "return" if (self.value is None) else ("return " + str(self.value))

    def accept(self, v: TACVisitor) -> None:
        v.visitReturn(self)


# Annotation (used for debugging).
class Memo(TACInstr):
    def __init__(self, msg: str) -> None:
        super().__init__(InstrKind.SEQ, [], [], None)
        self.msg = msg

    def __str__(self) -> str:
        return "memo '%s'" % self.msg

    def accept(self, v: TACVisitor) -> None:
        v.visitMemo(self)


# Label (function entry or branching target).
class Mark(TACInstr):
    def __init__(self, label: Label) -> None:
        super().__init__(InstrKind.LABEL, [], [], label)

    def __str__(self) -> str:
        return "%s:" % str(self.label)

    def accept(self, v: TACVisitor) -> None:
        v.visitMark(self)


# Parameter setting instruction.
class Param(TACInstr):
    def __init__(self, parameter: Temp) -> None:
        super().__init__(InstrKind.SEQ, [], [parameter], None)
        self.parameter = parameter

    def __str__(self) -> str:
        return 'param %s' % (self.parameter)

    def accept(self, v: TACVisitor) -> None:
        return v.visitParam(self)


# Calling a function.
class Call(TACInstr):
    def __init__(self, ret_v: Temp, target: Label, args: list[Temp]) -> None:
        super().__init__(InstrKind.SEQ, [ret_v], [], target)
        self.ret_v = ret_v
        self.target = target
        self.args = args

    def __str__(self) -> str:
        return '%s = call %s' % (self.ret_v, self.target)

    def accept(self, v: TACVisitor) -> None:
        return v.visitCall(self)


# Load address of symbol.
class LoadSymbol(TACInstr):
    def __init__(self, dst: Temp, symbolName: str) -> None:
        super().__init__(InstrKind.SEQ, [dst], [], None)
        self.dst = dst
        self.symbolName = symbolName

    def __str__(self) -> str:
        return '%s = load_symbol %s' % (self.dst, self.symbolName)

    def accept(self, v: TACVisitor) -> None:
        return v.visitLoadSymbol(self)


# Store data in address = base + offset.
class Store(TACInstr):
    def __init__(self, src: Temp, base: Temp, offset: int) -> None:
        super().__init__(InstrKind.SEQ, [], [src, base], None)
        self.src = src
        self.base = base
        self.offset = offset

    def __str__(self) -> str:
        return 'store %s, %s, %s' % (self.src, self.base, self.offset)

    def accept(self, v: TACVisitor) -> None:
        return v.visitStore(self)


# Load data in address = base + offset.
class Load(TACInstr):
    def __init__(self, dst: Temp, base: Temp, offset: int) -> None:
        super().__init__(InstrKind.SEQ, [dst], [base], None)
        self.dst = dst
        self.base = base
        self.offset = offset

    def __str__(self) -> str:
        return '%s = load %s, %s' % (self.dst, self.base, self.offset)

    def accept(self, v: TACVisitor) -> None:
        return v.visitLoad(self)

class Alloc(TACInstr):
    def __init__(self, dst: Temp, size: int) -> None:
        super().__init__(InstrKind.SEQ, [dst], [], None)
        self.dst = dst
        self.size = size

    def __str__(self) -> str:
        return '%s = alloc %s' % (self.dst, self.size)

    def accept(self, v: TACVisitor) -> None:
        return v.visitAlloc(self)