import React from 'react';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@mui/material';
import '../style/tablescale.css'

const TableScale = ({ data }) => {
  const dataObject = Object.values(data)[0];
  const tableData = dataObject.uploaded_data;
  const headers = tableData.length ? Object.keys(tableData[0]) : [];
  console.log("headers", headers);

  return (
    <Paper sx={{ width: '100%', overflow: 'hidden'}}>
    <TableContainer id='scrollbar1' sx={{ maxHeight: 440 }}>
      <Table stickyHeader sx={{ maxWidth: 650 }} aria-label="a dense table">
        <TableHead >
          <TableRow sx={{ '&:last-child td, &:last-child th': { border: 0 }, backgroundColor: 'ActiveBorder'}}>
            {/* <TableCell>File Name</TableCell> */}
            {headers.map((header) => (
              <TableCell sx={{fontWeight: 'bold'}} key={header}>{header}</TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {tableData.map((row, index) => (
            <TableRow key={index}>
              {/* <TableCell>
                    row.file_name : `Segment ${index + 1}`
                </TableCell>*/}
              {headers.map((header) => (
                <TableCell key={header}>{row[header]}</TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
    </Paper>
  );
};

export default TableScale;
