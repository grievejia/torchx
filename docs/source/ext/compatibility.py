#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import yaml
from docutils import nodes
from sphinx.util.docutils import SphinxDirective

COMPATIBILITY_SETS = {
    "scheduler": {
        "logs": "Fetch Logs",
        "distributed": "Distributed Jobs",
        "cancel": "Cancel Job",
        "describe": "Describe Job",
    },
}


class CompatibilityDirective(SphinxDirective):
    # this enables content in the directive
    has_content = True

    def run(self):
        """
        targetid = "todo-%d" % self.env.new_serialno("todo")
        targetnode = nodes.target("", "", ids=[targetid])
        todo_node = todo("\n".join(self.content))
        todo_node += nodes.title(_("Todo"), _("Todo"))
        self.state.nested_parse(self.content, self.content_offset, todo_node)
        if not hasattr(self.env, "todo_all_todos"):
            self.env.todo_all_todos = []

        self.env.todo_all_todos.append(
            {
                "docname": self.env.docname,
                "lineno": self.lineno,
                "todo": todo_node.deepcopy(),
                "target": targetnode,
            }
        )
        """

        raw_content = "\n".join(self.content)
        args = yaml.safe_load(raw_content)
        fields = COMPATIBILITY_SETS[args["type"]]
        features = args["features"]

        assert set(fields.keys()) == set(
            features.keys()
        ), f"features keys don't match expected {list(fields.keys())}"

        table = nodes.table("")
        group = nodes.tgroup("", cols=2)
        table.append(group)
        group.append(nodes.colspec("", colwidth=20))
        group.append(nodes.colspec("", colwidth=80))

        head = nodes.thead("")
        group.append(head)
        head_row = nodes.row()
        head.append(head_row)

        headers = ["Feature", "Scheduler Support"]
        for header in headers:
            entry = nodes.entry()
            entry += nodes.paragraph(text=header)
            head_row.append(entry)

        body = nodes.tbody("")
        group.append(body)

        for field, name in fields.items():
            row = nodes.row("")
            row.append(nodes.entry("", nodes.paragraph(text=name)))
            value = features[field]
            if value is True:
                text = "✔️"
            elif value is False:
                text = "❌"
            else:
                text = value
            row.append(nodes.entry("", nodes.paragraph(text=text)))
            body.append(row)

        return [
            table,
        ]


def setup(app):
    app.add_directive("compatibility", CompatibilityDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
