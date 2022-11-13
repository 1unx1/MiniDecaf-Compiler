import random

from backend.dataflow.basicblock import BasicBlock, BlockKind
from backend.dataflow.cfg import CFG
from backend.dataflow.loc import Loc
from backend.reg.regalloc import RegAlloc
from backend.riscv.riscvasmemitter import RiscvAsmEmitter
from backend.subroutineemitter import SubroutineEmitter
from backend.subroutineinfo import SubroutineInfo
from utils.riscv import Riscv
from utils.tac.holeinstr import HoleInstr
from utils.tac.reg import Reg
from utils.tac.temp import Temp

"""
BruteRegAlloc: one kind of RegAlloc

bindings: map from temp.index to Reg

we don't need to take care of GlobalTemp here
because we can remove all the GlobalTemp in selectInstr process

1. accept：根据每个函数的 CFG 进行寄存器分配，寄存器分配结束后生成相应汇编代码
2. bind：将一个 Temp 与寄存器绑定
3. unbind：将一个 Temp 与相应寄存器解绑定
4. localAlloc：根据数据流对一个 BasicBlock 内的指令进行寄存器分配
5. allocForLoc：每一条指令进行寄存器分配
6. allocRegFor：根据数据流决定为当前 Temp 分配哪一个寄存器
"""

class BruteRegAlloc(RegAlloc):
    def __init__(self, emitter: RiscvAsmEmitter) -> None:
        super().__init__(emitter)
        self.bindings = {}
        for reg in emitter.allocatableRegs:
            reg.used = False
        self.regIndex = 0 # index of allocatableRegs

    def accept(self, graph: CFG, info: SubroutineInfo, params: list[Temp]) -> None:
        subEmitter = self.emitter.emitSubroutine(info, params)
        for bb in graph.iterator():
            # you need to think more here
            # maybe we don't need to alloc regs for all the basic blocks
            if graph.unreachable(bb.id):
                continue
            if bb.label is not None:
                subEmitter.emitLabel(bb.label)
            self.localAlloc(bb, subEmitter)
        subEmitter.emitEnd()

    def bind(self, temp: Temp, reg: Reg):
        reg.used = True
        self.bindings[temp.index] = reg
        reg.occupied = True
        reg.temp = temp

    def unbind(self, temp: Temp):
        if temp.index in self.bindings:
            self.bindings[temp.index].occupied = False
            self.bindings.pop(temp.index)

    def localAlloc(self, bb: BasicBlock, subEmitter: SubroutineEmitter):
        self.bindings.clear()
        for reg in self.emitter.allocatableRegs:
            reg.occupied = False
        # the first basic block
        if bb.id == 0:
            for i, param in enumerate(subEmitter.params[:8]):
                self.bind(param, Riscv.ArgRegs[i])

        # in step9, you may need to think about how to store callersave regs here
        for loc in bb.allSeq():
            subEmitter.emitComment(str(loc.instr))
            if isinstance(loc.instr, Riscv.Call):
                self.allocForCall(loc, subEmitter)
            else:
                self.allocForLoc(loc, subEmitter)

        for tempindex in bb.liveOut:
            if tempindex in self.bindings:
                subEmitter.emitStoreToStack(self.bindings.get(tempindex))

        if (not bb.isEmpty()) and (bb.kind is not BlockKind.CONTINUOUS):
            self.allocForLoc(bb.locs[len(bb.locs) - 1], subEmitter)

    def allocForCall(self, loc: Loc, subEmitter: SubroutineEmitter):
        # pass the parameters by stack
        for arg in reversed(loc.instr.args):
            reg = self.allocRegFor(arg, True, loc.liveIn, subEmitter)
            # push the parameters
            subEmitter.emitNative(Riscv.SPAdd(-4))
            subEmitter.emitNative(Riscv.NativeStoreWord(reg, Riscv.SP, 0))
            # change offsets due to the change of SP
            subEmitter.changeOffset(4)
        # store caller save regs, which must be done after allocRegFor args' temps!
        callerSaveRegs = []
        for reg in self.emitter.callerSaveRegs:
            if reg.occupied and reg.temp.index in loc.liveOut:
                subEmitter.emitStoreToStack(reg)
                callerSaveRegs.append(reg)
        # pass the parameters by regs
        for i in range(len(loc.instr.args[:8])):
            # pop the parameters to arg regs
            subEmitter.emitNative(Riscv.NativeLoadWord(Riscv.ArgRegs[i], Riscv.SP, 0))
            subEmitter.emitNative(Riscv.SPAdd(4))
            # change offsets due to the change of SP
            subEmitter.changeOffset(-4)
        # call instr
        self.allocForLoc(loc, subEmitter)
        # if there are parameters passed by stack
        if len(loc.instr.args) > 8:
            size = 4 * (len(loc.instr.args) - 8)
            # free stack memory
            subEmitter.emitNative(Riscv.SPAdd(size))
            # change offsets due to the change of SP
            subEmitter.changeOffset(-size)
        # store return value
        self.saveReturnValue(loc, subEmitter)
        # load caller save regs
        for reg in callerSaveRegs:
            subEmitter.emitLoadFromStack(reg, reg.temp)

    def saveReturnValue(self, loc: Loc, subEmitter: SubroutineEmitter):
        tempForA0 = None
        if Riscv.A0.occupied:
            tempForA0 = Riscv.A0.temp
            self.unbind(Riscv.A0.temp)
        self.bind(loc.instr.ret_v, Riscv.A0)
        subEmitter.emitStoreToStack(Riscv.A0)
        self.unbind(Riscv.A0.temp)
        if tempForA0:
            self.bind(tempForA0, Riscv.A0)

    def allocForLoc(self, loc: Loc, subEmitter: SubroutineEmitter):
        instr = loc.instr
        srcRegs: list[Reg] = []
        dstRegs: list[Reg] = []

        for i in range(len(instr.srcs)):
            temp = instr.srcs[i]
            if isinstance(temp, Reg):
                srcRegs.append(temp)
            else:
                srcRegs.append(self.allocRegFor(temp, True, loc.liveIn, subEmitter))

        for i in range(len(instr.dsts)):
            temp = instr.dsts[i]
            if isinstance(temp, Reg):
                dstRegs.append(temp)
            else:
                dstRegs.append(self.allocRegFor(temp, False, loc.liveIn, subEmitter))

        subEmitter.emitNative(instr.toNative(dstRegs, srcRegs))

    def allocRegFor(
        self, temp: Temp, isRead: bool, live: set[int], subEmitter: SubroutineEmitter
    ):
        if temp.index in self.bindings:
            return self.bindings[temp.index]

        for reg in self.emitter.allocatableRegs:
            if (not reg.occupied) or (not reg.temp.index in live):
                subEmitter.emitComment(
                    "  allocate {} to {}  (read: {}):".format(
                        str(temp), str(reg), str(isRead)
                    )
                )
                if isRead:
                    subEmitter.emitLoadFromStack(reg, temp)
                if reg.occupied:
                    self.unbind(reg.temp)
                self.bind(temp, reg)
                return reg

        reg = self.emitter.allocatableRegs[self.regIndex]
        self.regIndex = (self.regIndex + 1) % len(self.emitter.allocatableRegs)
        subEmitter.emitStoreToStack(reg)
        subEmitter.emitComment("  spill {} ({})".format(str(reg), str(reg.temp)))
        self.unbind(reg.temp)
        self.bind(temp, reg)
        subEmitter.emitComment(
            "  allocate {} to {} (read: {})".format(str(temp), str(reg), str(isRead))
        )
        if isRead:
            subEmitter.emitLoadFromStack(reg, temp)
        return reg
