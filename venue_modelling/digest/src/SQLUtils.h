#pragma once
#include <string.h> 
#include <iostream>

#include "mysql_connection.h"

#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>

int selectInt(std::string sql, std::string valueName);

float selectFloat(std::string sql, std::string valueName);

std::string selectString(std::string sql, std::string valueName);

void ticketReady(std::string username);
