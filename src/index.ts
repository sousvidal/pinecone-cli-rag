#!/usr/bin/env node

import { Command } from "commander";
import { indexCommand } from "./commands/index.js";
import { queryCommand } from "./commands/query.js";
import { searchCommand } from "./commands/search.js";

const program = new Command();

program
  .name("rag")
  .description("An advanced RAG (Retrieval-Augmented Generation) CLI tool")
  .version("0.1.0");

program.addCommand(indexCommand);
program.addCommand(queryCommand);
program.addCommand(searchCommand);

program.parse(process.argv);
