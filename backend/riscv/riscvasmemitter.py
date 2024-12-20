from typing import Sequence, Tuple

from backend.asmemitter import AsmEmitter
from utils.error import IllegalArgumentException
from utils.label.label import Label, LabelKind
from utils.riscv import Riscv
from utils.tac.reg import Reg
from utils.tac.tacfunc import TACFunc
from utils.tac.tacinstr import *
from utils.tac.tacvisitor import TACVisitor

from ..subroutineemitter import SubroutineEmitter
from ..subroutineinfo import SubroutineInfo

"""
RiscvAsmEmitter: an AsmEmitter for RiscV
"""


class RiscvAsmEmitter(AsmEmitter):
    def __init__(
        self,
        allocatableRegs: list[Reg],
        callerSaveRegs: list[Reg],
        globalSymbolNameValues: dict[str, Union[int, list[int]]]
    ) -> None:
        super().__init__(allocatableRegs, callerSaveRegs)

    
        # the start of the asm code
        # int step10, you need to add the declaration of global var here
        if globalSymbolNameValues:
            self.printer.println('.data')
            for symbolName, initValue in globalSymbolNameValues.items():
                self.printer.println('.global %s' % (symbolName))
                self.printer.printLabel(Label(LabelKind.TEMP, symbolName))
                if isinstance(initValue, int): # global variable
                    self.printer.println('.word %s' % (initValue))
                else: # global array
                    if isinstance(initValue, list):
                        for integer in initValue:
                            self.printer.println('.word %s' % (integer))
                    else:
                        self.printer.println('.zero %s' % (initValue[1] * 4))
                self.printer.println("")
        self.printer.println(".text")
        self.printer.println(".global main")
        self.printer.println("")

    # transform tac instrs to RiscV instrs
    # collect some info which is saved in SubroutineInfo for SubroutineEmitter
    def selectInstr(self, func: TACFunc) -> tuple[list[str], SubroutineInfo]:

        selector: RiscvAsmEmitter.RiscvInstrSelector = (
            RiscvAsmEmitter.RiscvInstrSelector(func.entry)
        )
        for instr in func.getInstrSeq():
            instr.accept(selector)

        info = SubroutineInfo(func.entry)

        return (selector.seq, info)

    # use info to construct a RiscvSubroutineEmitter
    def emitSubroutine(self, info: SubroutineInfo, params: list[Temp], arrays: list[tuple[Temp, int]]):
        return RiscvSubroutineEmitter(self, info, params, arrays)

    # return all the string stored in asmcodeprinter
    def emitEnd(self):
        return self.printer.close()

    class RiscvInstrSelector(TACVisitor):
        def __init__(self, entry: Label) -> None:
            self.entry = entry
            self.seq = []

        # in step11, you need to think about how to deal with globalTemp in almost all the visit functions. 
        def visitReturn(self, instr: Return) -> None:
            if instr.value is not None:
                self.seq.append(Riscv.Move(Riscv.A0, instr.value))
            else:
                self.seq.append(Riscv.LoadImm(Riscv.A0, 0))
            self.seq.append(Riscv.JumpToEpilogue(self.entry))

        def visitMark(self, instr: Mark) -> None:
            self.seq.append(Riscv.RiscvLabel(instr.label))

        def visitLoadImm4(self, instr: LoadImm4) -> None:
            self.seq.append(Riscv.LoadImm(instr.dst, instr.value))

        def visitUnary(self, instr: Unary) -> None:
            self.seq.append(Riscv.Unary(instr.op, instr.dst, instr.operand))
 
        def visitBinary(self, instr: Binary) -> None:
            if instr.op == BinaryOp.EQU:
                self.seq.append(Riscv.Binary(BinaryOp.SUB, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(UnaryOp.SEQZ, instr.dst, instr.dst))
            elif instr.op == BinaryOp.NEQ:
                self.seq.append(Riscv.Binary(BinaryOp.SUB, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(UnaryOp.SNEZ, instr.dst, instr.dst))
            elif instr.op == BinaryOp.LEQ:
                self.seq.append(Riscv.Binary(BinaryOp.SGT, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(UnaryOp.SEQZ, instr.dst, instr.dst))
            elif instr.op == BinaryOp.GEQ:
                self.seq.append(Riscv.Binary(BinaryOp.SLT, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(UnaryOp.SEQZ, instr.dst, instr.dst))
            elif instr.op == BinaryOp.AND:
                self.seq.append(Riscv.Unary(UnaryOp.SNEZ, instr.lhs, instr.lhs))
                self.seq.append(Riscv.Unary(UnaryOp.SNEZ, instr.rhs, instr.rhs))
                self.seq.append(Riscv.Binary(BinaryOp.AND, instr.dst, instr.lhs, instr.rhs))
            elif instr.op == BinaryOp.OR:
                self.seq.append(Riscv.Binary(BinaryOp.OR, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(UnaryOp.SNEZ, instr.dst, instr.dst))
            else:
                self.seq.append(Riscv.Binary(instr.op, instr.dst, instr.lhs, instr.rhs))

        def visitCondBranch(self, instr: CondBranch) -> None:
            self.seq.append(Riscv.Branch(instr.cond, instr.label))
        
        def visitBranch(self, instr: Branch) -> None:
            self.seq.append(Riscv.Jump(instr.target))

        def visitAssign(self, instr: Assign) -> None:
            self.seq.append(Riscv.Move(instr.dst, instr.src))

        def visitParam(self, instr: Param) -> None:
            pass

        # in step9, you need to think about how to pass the parameters and how to store and restore callerSave regs
        def visitCall(self, instr: Call) -> None:
            self.seq.append(Riscv.Call(instr.ret_v, instr.target, instr.args))
        
        def visitLoadSymbol(self, instr: LoadSymbol) -> None:
            self.seq.append(Riscv.LoadAddress(instr.dst, instr.symbolName))

        def visitStore(self, instr: Store) -> None:
            self.seq.append(Riscv.Store(instr.src, instr.base, instr.offset))

        def visitLoad(self, instr: Load) -> None:
            self.seq.append(Riscv.Load(instr.dst, instr.base, instr.offset))

        # in step11, you need to think about how to store the array
        def visitAlloc(self, instr: Alloc) -> None:
            self.seq.append(Riscv.LoadArray(instr.dst))
"""
RiscvAsmEmitter: an SubroutineEmitter for RiscV
"""

class RiscvSubroutineEmitter(SubroutineEmitter):
    def __init__(self, emitter: RiscvAsmEmitter, info: SubroutineInfo, params: list[Temp], arrays: list[tuple[Temp, int]]) -> None:
        super().__init__(emitter, info)
        
        # + 4 is for the RA reg, + 4 is for the FP reg, so + 8
        self.nextLocalOffset = 4 * len(Riscv.CalleeSaved) + 8
        
        # the buf which stored all the NativeInstrs in this function
        self.buf: list[NativeInstr] = []

        # from temp to int
        # record where a temp is stored in the stack
        self.offsets = {}

        self.printer.printLabel(info.funcLabel)

        # in step9, step11 you can compute the offset of local array and parameters here
        # parameters of this function
        self.params = params
        # offset to SP of each local array of this function
        self.arraySPOffsets = {}
        for addrTemp, size in arrays:
            self.arraySPOffsets[addrTemp.index] = self.nextLocalOffset
            self.nextLocalOffset += size
        # offset to FP of each parameter of this function
        self.paramFPOffsets = {param.index : 4 * i for i, param in enumerate(self.params[8:])}

    def emitComment(self, comment: str) -> None:
        # you can add some log here to help you debug
        pass
    
    def changeOffset(self, delta: int):
        self.nextLocalOffset += delta
        for i in self.offsets:
            self.offsets[i] += delta
        for i in self.arraySPOffsets:
            self.arraySPOffsets[i] += delta

    # store some temp to stack
    # usually happen when reaching the end of a basicblock
    # in step9, you need to think about the fuction parameters here
    def emitStoreToStack(self, src: Reg) -> None:
        if src.temp.index not in self.offsets:
            self.offsets[src.temp.index] = self.nextLocalOffset
            self.nextLocalOffset += 4
        self.buf.append(
            Riscv.NativeStoreWord(src, Riscv.SP, self.offsets[src.temp.index])
        )

    # load some temp from stack
    # usually happen when using a temp which is stored to stack before
    # in step9, you need to think about the fuction parameters here
    def emitLoadFromStack(self, dst: Reg, src: Temp):
        if src.index not in self.offsets:
            if src.index not in self.paramFPOffsets:
                raise IllegalArgumentException()
            else:
                self.buf.append(Riscv.NativeLoadWord(dst, Riscv.FP, self.paramFPOffsets[src.index]))
        else:
            self.buf.append(
                Riscv.NativeLoadWord(dst, Riscv.SP, self.offsets[src.index])
            )

    # add a NativeInstr to buf
    # when calling the fuction emitEnd, all the instr in buf will be transformed to RiscV code
    def emitNative(self, instr: NativeInstr):
        self.buf.append(instr)

    def emitLabel(self, label: Label):
        self.buf.append(Riscv.RiscvLabel(label).toNative([], []))

    
    def emitEnd(self):
        self.printer.printComment("start of prologue")
        self.printer.printInstr(Riscv.SPAdd(-self.nextLocalOffset))

        # in step9, you need to think about how to store RA here
        # you can get some ideas from how to save CalleeSaved regs
        # save RA
        self.printer.printInstr(Riscv.NativeStoreWord(Riscv.RA, Riscv.SP, 4 + 4 * len(Riscv.CalleeSaved)))
        # save FP
        self.printer.printInstr(Riscv.NativeStoreWord(Riscv.FP, Riscv.SP, 4 * len(Riscv.CalleeSaved)))
        # update FP
        self.printer.printInstr(Riscv.FPUpdate(self.nextLocalOffset))

        for i in range(len(Riscv.CalleeSaved)):
            if Riscv.CalleeSaved[i].isUsed():
                self.printer.printInstr(
                    Riscv.NativeStoreWord(Riscv.CalleeSaved[i], Riscv.SP, 4 * i)
                )

        self.printer.printComment("end of prologue")
        self.printer.println("")

        self.printer.printComment("start of body")

        # in step9, you need to think about how to pass the parameters here
        # you can use the stack or regs

        # using asmcodeprinter to output the RiscV code
        for instr in self.buf:
            self.printer.printInstr(instr)

        self.printer.printComment("end of body")
        self.printer.println("")

        self.printer.printLabel(
            Label(LabelKind.TEMP, self.info.funcLabel.name + Riscv.EPILOGUE_SUFFIX)
        )
        self.printer.printComment("start of epilogue")

        for i in range(len(Riscv.CalleeSaved)):
            if Riscv.CalleeSaved[i].isUsed():
                self.printer.printInstr(
                    Riscv.NativeLoadWord(Riscv.CalleeSaved[i], Riscv.SP, 4 * i)
                )
        # reload FP
        self.printer.printInstr(Riscv.NativeLoadWord(Riscv.FP, Riscv.SP, 4 * len(Riscv.CalleeSaved)))
        # reload RA
        self.printer.printInstr(Riscv.NativeLoadWord(Riscv.RA, Riscv.SP, 4 + 4 * len(Riscv.CalleeSaved)))
        self.printer.printInstr(Riscv.SPAdd(self.nextLocalOffset))
        self.printer.printComment("end of epilogue")
        self.printer.println("")

        self.printer.printInstr(Riscv.NativeReturn())
        self.printer.println("")
