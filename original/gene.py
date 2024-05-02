class Gene:
    def __init__(self, g, tp, inode, onode):
        self.lnk = Link(tp, g.lnk.weight, inode, onode, g.lnk.is_recurrent)
        self.innovation_num = g.innovation_num
        self.mutation_num = g.mutation_num
        self.enable = g.enable

    def __init__(self, xline, traits, nodes):
        st = StringTokenizer(xline)
        inode = None
        onode = None
        for i in range(2):
            curword = st.nextToken()
        weight = Double(curword)
        recur = Integer(curword) == 1
        for i in range(3):
            curword = st.nextToken()
            innovation_num = Double(curword)
        for i in range(3):
            curword = st.nextToken()
            mutation_num = Double(curword)
        for i in range(2):
            curword = st.nextToken()
            enable = Integer(curword) == 1
        traitptr = None
        if len(traits) > 0 and traits is not None:
            for trait in traits:
                if trait.trait_id == trait_num:
                    traitptr = trait
                    break
        for i in range(len(nodes)):
            node = nodes[i]
            if node.node_id == inode_num:
                inode = node
            if node.node_id == onode_num:
                onode = node
        self.lnk = Link(traitptr, weight, inode, onode, recur)
        self.innovation_num = innovation_num
        self.mutation_num = mutation_num
        self.enable = enable

    def op_view(self):
        fmt03 = DecimalFormat("0.000;-0.000")
        fmt5 = DecimalFormat("0000")
        print("\n [Link (" + fmt5.format(self.lnk.in_node.node_id) + "," + fmt5.format(self.lnk.out_node.node_id) + "] innov (" + fmt5.format(self.innovation_num) + "\n")
        print("  mut=" + fmt03.format(self.mutation_num) + ") Weight " + fmt03.format(self.lnk.weight))
        if self.lnk.linktrait is not None:
            print(" Link's trait_id " + self.lnk.linktrait.trait_id)
        if not self.enable:
            print(" -DISABLED-")
        if self.lnk.is_recurrent:
            print(" -RECUR-")

    def __init__(self):
        pass

    def __init__(self, tp, w, inode, onode, recur, innov, mnum):
        self.lnk = Link(tp, w, inode, onode, recur)
        self.innovation_num = innov
        self.mutation_num = mnum
        self.enable = True

    def print_to_file(self, xFile):
        s2 = StringBuffer("")
        s2.append("gene ")
        if self.lnk.linktrait is not None:
            s2.append(" " + self.lnk.linktrait.trait_id)
        else:
            s2.append(" 0")
        s2.append(" " + self.lnk.in_node.node_id)
        s2.append(" " + self.lnk.out_node.node_id)
        s2.append(" " + self.lnk.weight)
        if self.lnk.is_recurrent:
            s2.append(" 1")
        else:
            s2.append(" 0")
        s2.append(" " + self.innovation_num)
        s2.append(" " + self.mutation_num)
        if self.enable:
            s2.append(" 1")
        else:
            s2.append(" 0")
        xFile.IOseqWrite(s2.toString())