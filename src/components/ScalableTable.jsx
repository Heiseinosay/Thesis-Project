import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@mui/material';

// const data = {
//   "RMS1": 0.0006827119505032897,
//   "RMS2": 0.0009519330924376845,
//   // Add all your JSON data here
//   "SpectralCentroid87": 1424.618680965721,
//   "ZCR87": 0.0048828125
// };

const ScalableTable = () => {
  const jsonData = Object.entries(data).map(([key, value]) => ({ key, value }));

  return (
    <div style={{ height: '400px', width: '100%' }}>
      <AutoSizer>
        {({ height, width }) => (
          <Table
            width={width}
            height={height}
            headerHeight={40}
            rowHeight={30}
            rowCount={jsonData.length}
            rowGetter={({ index }) => jsonData[index]}
          >
            <Column
              label="Parameter"
              dataKey="key"
              width={width * 0.5}
            />
            <Column
              label="Value"
              dataKey="value"
              width={width * 0.5}
            />
          </Table>
        )}
      </AutoSizer>
    </div>
  );
};

export default ScalableTable;
