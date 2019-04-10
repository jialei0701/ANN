class Perm:
    def __init__(self, group_name, index, buffer_name):
        self.group_name = group_name
        self.index = index
        self.buffer_name = buffer_name
        self.permutator_group3 = None

    def get_real_buffer_name(self, perm_inst):
        idx = perm_inst[self.group_name][self.index]
        return self.permutator_group3[self.group_name][idx][self.buffer_name]

    def get_all_buffer_names(self):
        return [idx[self.buffer_name] for idx in self.permutator_group3[self.group_name]]

    def set_permutator_group(self, permutator_group3):
        self.permutator_group3 = permutator_group3

    def is_same(self, that):
        equal = True

        equal = equal and self.group_name == that.group_name
        equal = equal and self.index == that.index
        equal = equal and self.buffer_name == that.buffer_name
        equal = equal and self.permutator_group3 == that.permutator_group3

        return equal



    def __str__(self):
        return "Perm(group={}, index={}, buffer={})".format(self.buffer_name, self.index, self.buffer_name)

